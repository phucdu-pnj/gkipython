from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker

# 1. Đọc và chuẩn bị dữ liệu
base_dir = Path(__file__).resolve().parent
file_name = base_dir / 'abc-28.csv'
df = pd.read_csv(file_name, header=None)
if df.shape != (365, 7):
    raise ValueError(f"Dữ liệu phải có 365 dòng và 7 cột; hiện có {df.shape[0]} dòng, {df.shape[1]} cột.")
df = df.apply(pd.to_numeric, errors='raise')
if not np.isfinite(df.to_numpy()).all():
    raise ValueError("Dữ liệu giá không được chứa giá trị thiếu hoặc vô hạn.")
day_count = len(df)
label_x = day_count + 13
candle_days = set()
for start in range(0, day_count, 30):
    end = min(start + 30, day_count)
    candle_count = min(8, max(1, round((end - start) * 8 / 30)))
    bucket_width = (end - start) / candle_count
    candle_days.update(np.linspace(start + bucket_width / 2,
                                   end - bucket_width / 2,
                                   candle_count, dtype=int))

# 7 cột tương ứng với các mức giá trong ngày
price_cols = [0, 1, 2, 3, 4, 5, 6]
df['Open'] = df[0]
df['Close'] = df[6]
df['High'] = df[price_cols].max(axis=1)
df['Low'] = df[price_cols].min(axis=1)

# Tính giá tham chiếu, giá trần, giá sàn
df['Ref_Price'] = df['Close'].shift(1)
df.loc[0, 'Ref_Price'] = df.loc[0, 'Open'] # Ngày đầu tiên giả sử tham chiếu = mở cửa
df['Floor_Price'] = df['Ref_Price'] * 0.9
df['Ceil_Price'] = df['Ref_Price'] * 1.1

# Lấy Min/Max toàn bộ dữ liệu để set limit cho trục Y
global_max = df['High'].max()
global_min = df['Low'].min()

# 2. Khởi tạo Figure và Subplots (Giao diện giống hình)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
fig.patch.set_facecolor('black')
plt.subplots_adjust(left=0.11, right=0.95, bottom=0.1, top=0.99, hspace=0.08)

for ax in [ax1, ax2]:
    ax.set_facecolor('black')
    ax.tick_params(colors='white', labelsize=12)
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.set_ylabel('VND', color='white', fontweight='bold')
    # Format trục Y theo chuẩn hàng nghìn (ví dụ: 20,000)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

y_axis_max = np.ceil(global_max / 5000) * 5000
for ax in [ax1, ax2]:
    ax.set_ylim(0, y_axis_max)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5000))

ax2.set_xlabel('Date', color='white', fontweight='bold')
ax2.set_xticks(np.arange(0, day_count, 30))
ax2.set_xlim(-5, day_count + 35)

# Vẽ các đường tham chiếu ngang cố định trên ax2 (Giống nét đứt xanh/đỏ trong hình)
ax2.axhline(y=global_max, color='green', linestyle='--', lw=0.8, alpha=0.6)
ax2.axhline(y=global_min, color='red', linestyle='--', lw=0.8, alpha=0.6)
ax2.text(label_x, global_max, f"{int(global_max):,}", color='green', va='center', ha='right')
ax2.text(label_x, global_min, f"{int(global_min):,}", color='red', va='center', ha='right')

# 3. Các biến dùng cho animation
days_data = []
lows_data = []
highs_data = []

# Đối tượng đồ họa sẽ được update
line_current = ax2.axhline(y=0, color='brown', linestyle='-', lw=1.5, alpha=0.7)
text_current = ax2.text(label_x, 0, "", color='red', va='center', ha='right')
fill_collection = None # Dùng để fill dải màu (area chart) ở ax2
high_line, = ax2.plot([], [], color='olivedrab', lw=1.4)
low_line, = ax2.plot([], [], color='darkorange', lw=1.4)

def get_candle_color(close_p, ref_p, floor_p, ceil_p):
    """Xác định màu sắc theo yêu cầu đề bài"""
    # Dùng np.isclose để tránh sai số dấu phẩy động
    if np.isclose(close_p, floor_p, atol=1e-2): 
        return 'dodgerblue'   # Giá sàn (Xanh dương)
    if np.isclose(close_p, ceil_p, atol=1e-2): 
        return 'mediumorchid' # Giá trần (Tím)
    if np.isclose(close_p, ref_p, atol=1e-2): 
        return 'yellow'       # Giá tham chiếu (Vàng)
    if close_p < ref_p: 
        return 'red'          # Giảm (Đỏ)
    return 'limegreen'        # Tăng (Xanh lá)

def init():
    return []

def animate(i):
    global fill_collection
    if i >= len(df): return []
    
    row = df.iloc[i]
    day = i
    o, c, h, l = row['Open'], row['Close'], row['High'], row['Low']
    ref, floor, ceil = row['Ref_Price'], row['Floor_Price'], row['Ceil_Price']
    
    color = get_candle_color(c, ref, floor, ceil)
    
    # ---------------------------------------------
    # BIỂU ĐỒ TRÊN (ax1) - CANDLESTICK
    # ---------------------------------------------
    if day in candle_days:
        # 1. Vẽ râu nến (High - Low)
        ax1.plot([day, day], [l, h], color=color, lw=1.8)

        # 2. Vẽ thân nến (Open - Close)
        body_bottom = min(o, c)
        body_height = max(abs(o - c), (global_max - global_min) * 0.005) # Cấp độ dày tối thiểu nếu nến Doji

        rect = plt.Rectangle((day - 1.1, body_bottom), 2.2, body_height,
                             facecolor=color, edgecolor=color)
        ax1.add_patch(rect)

        # Vẽ vạch trắng nhỏ nằm ngang giữa thân nến (tạo độ nổi khối giống hình mẫu)
        ax1.plot([day - 0.8, day + 0.8], [(o+c)/2, (o+c)/2], color='white', lw=0.9, alpha=0.8)

    # ---------------------------------------------
    # BIỂU ĐỒ DƯỚI (ax2) - AREA BAND THÚC ĐẨY THEO THỜI GIAN
    # ---------------------------------------------
    days_data.append(day)
    lows_data.append(l)
    highs_data.append(h)
    
    # Cập nhật Area (Dải băng giá dao động giống hình mẫu)
    if fill_collection:
        fill_collection.remove()
    fill_collection = ax2.fill_between(days_data, lows_data, highs_data,
                                       color='darkorange', alpha=0.45)
    high_line.set_data(days_data, highs_data)
    low_line.set_data(days_data, lows_data)
    
    # Cập nhật đường báo giá hiện tại (Nét ngang đỏ sậm)
    line_current.set_ydata([c, c])
    text_current.set_position((label_x, c))
    text_current.set_text(f"{int(c):,}")
    
    return []

# 4. Chạy mô phỏng Animation
playback_fps = day_count / 180
ani = animation.FuncAnimation(fig, animate, frames=day_count, init_func=init,
                              interval=1000 / playback_fps, blit=False)

# Để lưu thành video mp4 (yêu cầu máy có cài ffmpeg):
try:
    writer = animation.FFMpegWriter(fps=playback_fps)
    ani.save('dothi.mp4', writer=writer)
    print("Đã xuất file video: dothi.mp4")
except:
    print("Không tìm thấy ffmpeg. Chuyển sang hiển thị trực tiếp (Live Window).")
    plt.show()