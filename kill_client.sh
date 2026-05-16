#!/bin/sh
pkill -f 'org.prismlauncher.EntryPoint' && sleep 3 && pkill -9 -f 'org.prismlauncher.EntryPoint' 2>/dev/null
exit 0
