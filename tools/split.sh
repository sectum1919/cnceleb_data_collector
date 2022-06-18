if [ ! -d $3 ];then
 mkdir $3
fi

for name in `ls $2`
do
 if [ ! -d $3/$name ];then
  mkdir $3/$name
 fi

 for type in `ls $2/$name`
 do
  if [ ! -d $3/$name/$type ];then
   mkdir $3/$name/$type
  fi
  for txt in `find $2/$name/$type -name "*.txt"`
  do
   IFS=$'\n'
   finalname0=${txt##*/}
   finalname=${finalname0%.*}
   i=1
   input=$1/$name/$type/$finalname".mp4"
   `ffprobe $input 2> ./fps_verify.txt`
   fps=`grep -c "25 fps" ./fps_verify.txt`
   if [ $fps != 1 ];then
     echo "${input} is not 25 fps"
     continue
   fi

   for line in `cat $txt`
   do
    if [ ${line:0:1} == "=" ];then
      continue
    fi

    btime0=`echo $line | cut -f 1`
    b=${btime0##*:} # content after ":"
    # echo $b
    b=$[10#$b] # str to number
    # echo $b
    ((bmm=b*40)) # frame to time
    btime=${btime0%:*}"."${bmm}
    # echo $btime

    ltime0=`echo $line | cut -f 2`
    l=${ltime0##*:}
    l=$[10#$l]
    ((lmm=l*40))
    ltime=${ltime0%:*}"."${lmm}

    echo $3"/"$name"/"$type"/"$finalname"-"$i".mp4"

    ffmpeg -v quiet -ss $btime -t $ltime -accurate_seek -i $input -avoid_negative_ts 1 -strict -2 -y $3"/"$name"/"$type"/"$finalname"-"$i".mp4"
    let i++
   #`python3 split.py $input $outputdir $btime $ttime > splitpy.out`
   done
  done
 done
done
echo -e "\033[31mcomplete\033[0m"
