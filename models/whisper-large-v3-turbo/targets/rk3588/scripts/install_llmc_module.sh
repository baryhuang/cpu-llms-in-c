#!/bin/sh
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "error: run as root" >&2
    exit 1
fi

if [ "$#" -ne 1 ]; then
    echo "usage: $0 /path/to/patched/rocket.ko" >&2
    exit 1
fi

module=$1
kernel=$(uname -r)
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
config=$script_dir/../config/rocket-llmc.conf
destination=/lib/modules/$kernel/updates/rocket.ko

test -f "$module"
test -f "$config"

case "$(modinfo -F vermagic "$module")" in
    "$kernel "*) ;;
    *)
        echo "error: $module was not built for running kernel $kernel" >&2
        exit 1
        ;;
esac

install -D -m 0644 "$module" "$destination"
install -m 0644 "$config" /etc/modprobe.d/rocket-llmc.conf
depmod -a "$kernel"

if grep -q '^rocket ' /proc/modules; then
    modprobe -r rocket
fi
modprobe rocket

test -c /dev/accel/accel0
test "$(cat /sys/module/rocket/parameters/rocket_npu_clk_hz)" = 600000000
printf 'installed %s\n' "$destination"
printf 'rocket_npu_clk_hz=%s\n' "$(cat /sys/module/rocket/parameters/rocket_npu_clk_hz)"
