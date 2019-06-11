
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.sql.SparkSession

object RecallRank {
  val moviesFeature2="/movie/data/rank/movie2/"
  val userFeature2="/movie/data/rank/user2/"

  val ratingTrain="/movie/data/recall/rating/train/"
  val ratingNewData = "/movie/data/rating/new/"

  val modelRank="/movie/model/rank/"

  //临时画像存贮路径
  //用完以后可以删掉
  //hdfs 通配符读取目录并删掉  资料
  // https://blog.csdn.net/SunnyYoona/article/details/53786397
  //globStatus
  val streamingData = "/streaming2/*/"

  var spark : SparkSession = null
  var fileSystem : FileSystem = null

  def main(args:Array[String])={
    spark = SparkSession.builder().getOrCreate()

    val movieFeatureDF = spark.read.format("parquet").option("header","true").load(moviesFeature2)
    val userFeatureDF = spark.read.format("parquet").option("header","true").load(userFeature2)
    val trainRatingDF = spark.read.format("parquet").option("header","true").load(ratingTrain)
    val LrModel = LinearRegressionModel.load(modelRank)

    //先召回，读streaming里面的数据进行召回
    // 如果streaming里面有数据，说明，用户有动静，有新的画像，需要进行重新推荐
    // 如果streaming里面没有该用户的数据，说明是冷启动或者用户没有新画像，
        // 如果是冷启动，直接读top就好了
        // 如果是没有动静，也无需进行重新推荐
    //那么，我们就监控/streaming2/目录啦
    fileSystem = FileSystem.get(new Configuration())

    val files = fileSystem.globStatus(new Path(streamingData))
    //val file = files{0}
    //    val path = file.getPath.toString
    //    val df = spark.read.format("parquet").option("header","false").load(path+"/*.parquet")

    for (file <- files){
      // val file = files{0}
      val path = file.getPath
      val pathS = path.toString
      // val path = file.getPath
      // val pathS = path.toString

      var streamingRatingDF = spark.read.format("parquet").option("header","false").load(pathS+"/*.parquet")
      var ratingNewDataDF = spark.emptyDataFrame
      if (fileSystem.exists(new Path(ratingNewData))){
        ratingNewDataDF = spark.read.format("parquet").option("header","true").load(ratingNewData)
      }

      //streaming中有新动静，我们才重新更新推荐列表
      val count = streamingRatingDF.count()
      if (0 != count){
        // 把这个rating追加到ratingNewData数据中
        streamingRatingDF.write.format("parquet").option("header","true").mode("append").save(ratingNewData)
        if (0 != ratingNewDataDF.count()){
          ratingNewDataDF = ratingNewDataDF.union(streamingRatingDF)
        }else{
          ratingNewDataDF = streamingRatingDF
        }

        //遍历df中的每一个userID，对每一个userID做召回和排序，再把结果写入推荐列表中
        val portrait = ratingNewDataDF.rdd.collect().toList
        for (record <- portrait){
          //val record = portrait(0)
          //scala> val record = portrait(0)
          //record: org.apache.spark.sql.Row = [2,457,4.0,1558448400]
          //这样转了以后都是Any类型
          val userId = record(0)
          val movieId = record(1)
          val rating = record(2)
          val timestamp = record(3)
          val movieRecall = RecallMethod.get_alike_movie_from_one(spark,
            movieFeatureDF,
            trainRatingDF,
            ratingNewDataDF,
            userId.asInstanceOf[Long],
            rating.asInstanceOf[Double],
            movieId.asInstanceOf[Long])

          //组装成datafame
          val userMovieRecall = RecallMethod.constructUserMovie2DataFrame(spark,userId.asInstanceOf[Long],movieRecall)

          //放到排序模型中去排序
          val userMovieRank = RankMethod.rank(spark,LrModel,userMovieRecall,userFeatureDF,movieFeatureDF)

          //取topK
          val userMovieRankTopK = RankMethod.getTopKFromDataFrame(spark,userMovieRank )

          //写入数据库
          RankMethod.write2DB(userMovieRankTopK, "192.168.199.227", "3306", "dataframe")
        }


      }
      ////删除该路径上的文件，如果该文件是一个目录的话，则递归删除该目录
      fileSystem.delete(path,true)
    }



    //再排序


  }

  //删除通配符目录下的文件
  def deleteFilesInWildDir(fileSystem : FileSystem, wildDir: String)={

    //
    //val fileSystem = FileSystem.get(new Configuration())
    //val files = fileSystem.globStatus(new Path(streamingData))
    val files = fileSystem.globStatus(new Path(wildDir))
    for (file <- files){
      //scala>     val path = file.getPath.toString
      //path: String = hdfs://foo-1.example.com:8020/streaming2/1558448354
      val path = file.getPath
      println(path.toString)
      //删除该路径上的文件，如果该文件是一个目录的话，则递归删除该目录
      fileSystem.delete(path,true)
    }
    }

  //针对单个用户
  //读取短期画像，根据用户最近看的电影类型进行召回
  //短期画像对某类电影评分高，则召回该类型电影
  //短期画像对某类电影评分低，则召回非该类型电影
  def myRecall()={

  }

  //加载rank模型，进行精排
  def myRank(): Unit ={

  }

}
