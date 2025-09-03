import psutil

def display_ram_usage():
    # Get virtual memory statistics
    ram = psutil.virtual_memory()

    # Convert bytes to gigabytes for better readability
    total_ram_gb = round(ram.total / (1024**3), 2)
    available_ram_gb = round(ram.available / (1024**3), 2)
    used_ram_gb = round(ram.used / (1024**3), 2)

    print(f"Total RAM: {total_ram_gb} GB")
    print(f"Available RAM: {available_ram_gb} GB")
    print(f"Used RAM: {used_ram_gb} GB")
    print(f"RAM Usage Percentage: {ram.percent}%")