beeline_url="jdbc:hive2://$1/default;principal=hive/_HOST@DF.SBRF.RU"
#pklis-ldrb00058.labiac.df.sbrf.ru:10000
log_file_hdfs_addr="log_file_addr.txt"
bash_log='local_bash_log.txt'
file=./SUCCESS_
attempts=0

function check_last_dm_partition {
  local query='"show partitions sklod_dwh_sbx_retail_mp_ext.x_ft_clnt_aggr_mnth"'
  local cmd="beeline --silent=true --showHeader=false --outputformat=tsv2 -u '$beeline_url' -e $query"
  local bl_cmd=`eval "$cmd | tail -1"`
  local last_date="$(cut -d '=' -f2 <<< "$bl_cmd")"
  local last_day_pr_m=`(date --date="$(date +'%Y-%m-01') - 1 second" +%s)`
  local last_day_pr_m=`date -d @$last_day_pr_m +%Y-%m-%d`
  if [[ "$last_date" == "$last_day_pr_m" ]]; then
    echo true
  else
    echo false
  fi
}

function drop_file_if_exists {
  for var in "$@"; do
    if test -f $var; then
      rm $var
    fi
  done
}

drop_file_if_exists $log_file_hdfs_addr $bash_log $file

touch $bash_log

while ! $(check_last_dm_partition) && (( attempts < 40 )); do
  echo `date +'%Y-%m-%d %H:%M:%S'` "last slice date not ready yet" >> $bash_log
  attempts=$((attempts+1))
  sleep 12h
done

attempts=0

while ! test -f "$file" && (( attempts < 10 )); do
  /opt/venvs/anaconda/bin/python ./main.py --beeline_node_and_port $1
  if test -f $log_file_hdfs_addr; then
    scores_log_agr=`cat $log_file_hdfs_addr`
    failed_models=`hdfs dfs -cat $scores_log_agr | grep PROBLEMS | tail -1`
    failed_models_cnt=`expr length "$failed_models"`
      
    if (( failed_models_cnt > 10 )); then
      /opt/venvs/anaconda/bin/python ./main.py --beeline_node_and_port $1
    fi
  fi
  attempts=$((attempts+1))
  echo `date +'%Y-%m-%d %H:%M:%S'` "$attempts scoring attempts made" >> $bash_log
  sleep 1h
done

drop_file_if_exists $file