import csv
import random
import os

# Set random seed for reproducibility of train/test split
random.seed(42)

# File paths
workspace_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(workspace_dir, "Pendataan Seberapa Sering Penggunaan dan Lama pemakaian Gadget Pada Zaman Sekarang (Responses) - Cleaning.csv")

encoded_file = os.path.join(workspace_dir, "encoded_data.csv")
data_gadget_file = os.path.join(workspace_dir, "data_gadget.csv")
train_file = os.path.join(workspace_dir, "train.csv")
test_file = os.path.join(workspace_dir, "test.csv")

# Mappings for categorical and ordinal variables

# 1. Rata-rata screen time perhari -> Midpoint/representative values
screen_time_map = {
    '< 2 jam': 1.0,
    '2-4 jam': 3.0,
    '4-6 jam': 5.0,
    '6-8 jam': 7.0,
    '> 8 jam': 10.0
}

# 2. Frekuensi membuka HP perhari -> Midpoint/representative values
hp_freq_map = {
    '< 20': 10.0,
    '20 - 50': 35.0,
    '50 - 100': 75.0,
    '> 100': 120.0
}

# 3. Jumlah aplikasi media sosial aktif -> Midpoint/representative values
app_count_map = {
    '1-2': 1.5,
    '3-4': 3.5,
    '5-6': 5.5,
    '> 7': 8.5
}

# 4. Durasi waktu tidur perhari -> Midpoint/representative values
sleep_dur_map = {
    '< 3 jam': 2.0,
    '3-5 jam': 4.0,
    '5-7 jam': 6.0,
    '> 8 jam': 9.0
}

# 5. Frekuensi ordinal (Tidur larut, Sering membuka sosmed saat belajar)
ordinal_frequency_map = {
    'Tidak pernah': 0,
    'Jarang': 1,
    'Sering': 2,
    'Sering sekali': 3
}

# 6. Sulit fokus tanpa mengecek HP (Ordinal)
difficulty_focus_map = {
    'Sangat mudah': 0,
    'Mudah': 1,
    'Sedang': 2,
    'Sulit': 3,
    'Sulit sekali': 4
}

# 7. Yes/No binary mappings
binary_map = {
    'Iya': 1,
    'Tidak': 0
}

# 8. Target: Tingkat kecanduan media sosial Anda
target_map = {
    'Rendah': 0,
    'Sedang': 1,
    'Tinggi': 2
}

# Unique social media platforms for one-hot encoding
social_media_platforms = [
    'Instagram', 'Tiktok', 'Youtube', 'X_twitter', 'Facebook', 'Threads', 'WhatsApp'
]

# Unique statuses for one-hot encoding
statuses = ['Mahasiswa', 'Pelajar', 'Pekerja']

# Helper to parse sleep time string to decimal hour
def parse_sleep_time(val):
    val = val.strip()
    if val == "22":
        return 22.0
    
    # Parse HH:MM format
    if ":" in val:
        parts = val.split(":")
        try:
            h = float(parts[0])
            m = float(parts[1])
            return h + (m / 60.0)
        except ValueError:
            pass
    try:
        return float(val)
    except ValueError:
        return 0.0

# Helper to parse focus hours (handling range values like '2 -5')
def parse_focus_hours(val):
    val = val.strip()
    if '-' in val:
        parts = val.split('-')
        try:
            val1 = float(parts[0].strip())
            val2 = float(parts[1].strip())
            return (val1 + val2) / 2.0
        except ValueError:
            pass
    try:
        return float(val)
    except ValueError:
        return 0.0

# Helper to clean and split social media string
def clean_social_media(val):
    cleaned = []
    # Split by comma
    parts = val.split(',')
    for part in parts:
        part = part.strip()
        # Case corrections and space cleaning
        part_lower = part.lower()
        if 'instagram' in part_lower:
            cleaned.append('Instagram')
        elif 'tiktok' in part_lower:
            cleaned.append('Tiktok')
        elif 'youtube' in part_lower:
            cleaned.append('Youtube')
        elif 'x/twitter' in part_lower or 'twitter' in part_lower:
            cleaned.append('X_twitter')
        elif 'facebook' in part_lower:
            cleaned.append('Facebook')
        elif 'threads' in part_lower:
            cleaned.append('Threads')
        elif 'whatsapp' in part_lower:
            cleaned.append('WhatsApp')
    return cleaned

# Read original file
with open(input_file, mode='r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

# Prepare encoded header
encoded_header = [
    'Usia',
    'Rata_rata_screen_time_perhari',
    'Frekuensi_membuka_HP_perhari',
    'Jumlah_aplikasi_media_sosial_aktif_yang_digunakan',
    'Durasi_waktu_tidur_perhari',
    'Jam_tidur_hours',
    'Tidur_larut_karena_sosial_media',
    'Jam_waktu_belajar_bekerja_fokus_sehari',
    'Sulit_fokus_tanpa_mengecek_HP',
    'Sering_membuka_media_sosial_saat_belajar_bekerja',
    'Merasa_cemas_jika_jauh_dari_HP',
    'Sering_membuka_sosial_media_tanpa_tujuan_jelas',
    'Penggunaan_media_sosial_mengganggu_belajar_kerja',
    'Pernah_mencoba_mengurangi_penggunaan_media_sosial_namun_gagal'
]

# Add one-hot encoded columns for Status
for status in statuses:
    encoded_header.append(f'Status_{status}')

# Add one-hot encoded columns for Social Media
for platform in social_media_platforms:
    encoded_header.append(f'Sosmed_{platform}')

# Add target column
encoded_header.append('Tingkat_kecanduan_media_sosial_Anda')

encoded_rows = []

# Process each row
for row in rows:
    usia = int(row[0])
    status_val = row[1].strip()
    screen_time = screen_time_map[row[2].strip()]
    hp_freq = hp_freq_map[row[3].strip()]
    app_count = app_count_map[row[4].strip()]
    
    # Clean social media
    active_platforms = clean_social_media(row[5])
    
    sleep_dur = sleep_dur_map[row[6].strip()]
    sleep_time = parse_sleep_time(row[7])
    sleep_late = ordinal_frequency_map[row[8].strip()]
    focus_hours = parse_focus_hours(row[9])
    difficulty_focus = difficulty_focus_map[row[10].strip()]
    open_social_media = ordinal_frequency_map[row[11].strip()]
    
    anxious = binary_map[row[12].strip()]
    no_purpose = binary_map[row[13].strip()]
    disruptive = binary_map[row[14].strip()]
    try_reduce_failed = binary_map[row[15].strip()]
    
    target = target_map[row[16].strip()]
    
    # Build encoded row
    encoded_row = [
        usia,
        screen_time,
        hp_freq,
        app_count,
        sleep_dur,
        sleep_time,
        sleep_late,
        focus_hours,
        difficulty_focus,
        open_social_media,
        anxious,
        no_purpose,
        disruptive,
        try_reduce_failed
    ]
    
    # One-hot encode Status
    for status in statuses:
        encoded_row.append(1 if status_val == status else 0)
        
    # One-hot encode Social Media
    for platform in social_media_platforms:
        encoded_row.append(1 if platform in active_platforms else 0)
        
    # Append target
    encoded_row.append(target)
    
    encoded_rows.append(encoded_row)

# Save the full encoded dataset
with open(encoded_file, mode='w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(encoded_header)
    writer.writerows(encoded_rows)

with open(data_gadget_file, mode='w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(encoded_header)
    writer.writerows(encoded_rows)

print(f"Dataset fully encoded and saved to: {encoded_file} and {data_gadget_file}")

# Train-test split (80% train, 20% test)
# Shuffle rows first using seed (to maintain split consistency)
random.shuffle(encoded_rows)

split_idx = int(len(encoded_rows) * 0.8)
train_rows = encoded_rows[:split_idx]
test_rows = encoded_rows[split_idx:]

# Save train set
with open(train_file, mode='w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(encoded_header)
    writer.writerows(train_rows)

# Save test set
with open(test_file, mode='w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(encoded_header)
    writer.writerows(test_rows)

print(f"Train/Test split completed:")
print(f"  - Train dataset saved to: {train_file} ({len(train_rows)} rows)")
print(f"  - Test dataset saved to: {test_file} ({len(test_rows)} rows)")
print("Preprocessing completed successfully!")
