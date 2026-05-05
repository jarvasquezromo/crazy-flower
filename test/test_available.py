import cflib.crtp

cflib.crtp.init_drivers()

# Scan specifically on your expected URI
available = cflib.crtp.scan_interfaces(address=0xE7E7E7E705)
# available = cflib.crtp.scan_interfaces()
print("Crazyflies found:")
for i in available:
    print("    >", i[0])