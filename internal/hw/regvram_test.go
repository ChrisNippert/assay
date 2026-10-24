package hw

import "testing"

func TestParseRegVRAM(t *testing.T) {
	out := `
HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0000
    DriverDesc    REG_SZ    Intel(R) UHD Graphics 770
    HardwareInformation.qwMemorySize    REG_QWORD    0x8000000

HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0001
    DriverDesc    REG_SZ    AMD Radeon RX 6700 XT
    HardwareInformation.qwMemorySize    REG_QWORD    0x300000000

HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\Properties
`
	name, v := parseRegVRAM(out)
	if name != "AMD Radeon RX 6700 XT" || v != 12<<30 {
		t.Errorf("got %q %d", name, v)
	}
	if name, v := parseRegVRAM(""); name != "" || v != 0 {
		t.Errorf("empty: %q %d", name, v)
	}
}
