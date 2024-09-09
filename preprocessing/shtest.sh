while :; do
	case $1 in
		-a|--flag1) echo "$2"; shift
		;;
		-b|--flag2) echo "$2"; shift
		;;
		-c|--optflag1) echo "-c"
		;;
		-d|--optflag2) echo "-d"
		;;
		-e|--optflag3) echo "-e"
		;;
		*) break
	esac
	shift
done
