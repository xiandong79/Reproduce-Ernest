#!/bin/bash
bin=`dirname "$0"`
bin=`cd "$bin"; pwd`
DIR=`cd $bin/../; pwd`
. "${DIR}/../bin/config.sh"
. "${DIR}/bin/config.sh"

echo "========== running ${APP} bench =========="


# pre-running
DU ${INPUT_HDFS} SIZE

JAR="${DIR}/target/KMeansApp-1.0.jar"
CLASS="KmeansApp"


setup
set_gendata_opt

function run_kmeans{

  /root/ephemeral-hdfs/bin/hadoop dfs -rm -r /SparkBench/KMeans/Output*

  OPTION=" ${INOUT_SCHEME}${INPUT_HDFS} ${INOUT_SCHEME}${OUTPUT_HDFS} ${NUM_OF_CLUSTERS} ${MAX_ITERATION} ${NUM_RUN} $scale $mcs"

  mcs=$1
  scale=$2
  echo -n "Cores $mcs "

  purge_data "${MC_LIST}"
  echo_and_run sh -c " ${SPARK_HOME}/bin/spark-submit --class $CLASS --total-executor-cores $mcs --master ${APP_MASTER} ${YARN_OPT} ${SPARK_OPT} ${SPARK_RUN_OPT} $JAR ${OPTION} 2>&1|grep "KmeansApp.*took" "

}
done
teardown

run_kmeans 2 0.015625
run_kmeans 2 0.021382
run_kmeans 32 0.125
run_kmeans 4 0.015625
run_kmeans 12 0.050164
run_kmeans 10 0.038651
run_kmeans 30 0.119243
run_kmeans 12 0.055921
run_kmeans 10 0.044408
run_kmeans 12 0.061678
run_kmeans 14 0.055921

exit 0
