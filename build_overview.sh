#!/usr/bin/env sh
########################################################################
# Copyright (C) 2025 Xu Ruijun
#
# Copying and distribution of this script, with or without modification,
# are permitted in any medium without royalty provided the copyright
# notice and this notice are preserved.  This file is offered as-is,
# without any warranty.
########################################################################
count_skip=0
count_ov=0

for file in "$@"; do
	if [ -f "$file.ovr" ]; then
		echo "$file" already with overview
		count_skip=`expr $count_skip + 1`
	else
		echo "$file" build overview
		gdaladdo -r average -ro "$file"
		count_ov=`expr $count_ov + 1`
	fi
done

echo $count_skip files skipped
echo $count_ov files build overview
