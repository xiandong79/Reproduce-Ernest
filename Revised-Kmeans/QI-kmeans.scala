/*
 * (C) Copyright IBM Corp. 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import org.apache.spark.internal.Logging
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

object KmeansApp {
  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN);
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF);
    if (args.length < 4) {
      println("usage: <input> <output> <numClusters> <maxIterations> <runs> - optional")
      System.exit(0)
    }
    val conf = new SparkConf
    conf.setAppName("Spark KMeans Example").set("spark.eventLog.enabled","true")
    val sc = new SparkContext(conf)

    val input = args(0)
    val output = args(1)
    val K = args(2).toInt
    val maxIterations = args(3).toInt
    val runs = calculateRuns(args)
    val sample_frac = args(5).toInt

    // val sampledData = sc.textFile(input)
    // var start = System.currentTimeMillis();
    // val loadTime = (System.currentTimeMillis() - start).toDouble / 1000.0
    val data = sc.textFile(input)
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()
    val sampledData = parsedData.sample(withReplacement=False, fraction=sample_frac).coalesce(numPartitions=32).cache()


    var start = System.currentTimeMillis();
    val clusters: KMeansModel = KMeans.train(sampledData, K, maxIterations, runs, KMeans.K_MEANS_PARALLEL, seed = 127L)
    println("cluster centers: " + clusters.clusterCenters.mkString(","))


    val vectorsAndClusterIdx = sampledData.map { point =>
      val prediction = clusters.predict(point)
      (point.toString, prediction)
    }
    vectorsAndClusterIdx.saveAsTextFile(output)

    val WSSSE = clusters.computeCost(sampledData)
    var end = System.currentTimeMillis();

    print "KmeansApp sample: ", sample_frac, " took ", (end-start).toDouble / 1000.0
    sc.stop()
  }

  def calculateRuns(args: Array[String]): Int = {
    if (args.length > 4) args(4).toInt
    else 1
  }
}
