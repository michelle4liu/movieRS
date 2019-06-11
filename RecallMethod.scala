import org.apache.spark.rdd
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object RecallMethod {
  //针对每个用户获取一个待排序的列表，获取同一类型的电影，100部
  def get_alike_movie_from_one(spark : SparkSession,
                               movieFeatureDF : DataFrame,
                               trainRatingDF : DataFrame,
                               streamingRatingDF : DataFrame,
                               userId:Long,
                               rating : Double,
                               movieId : Long): Array[Any] ={

    //首先获取user已经评价过的电影
    val movie_train_exist = trainRatingDF.filter(s"userId == ${userId}")
    val movie_streaming_exist = streamingRatingDF.filter(s"userId == ${userId}")

    //val movieIds1 = movie_train_exist.rdd.map{case Row(x : ResultVector)=>x.movieId}
    val movieIds1 = movie_train_exist.rdd.map(x=>x(1))
    val movieIds2 = movie_streaming_exist.rdd.map(x=>x(1))
    //里面的格式全部是Any哦
    //Array[Any] = Array(2340, 1028, 2804, 48, 720, 608, 260)
    val movieIds = movieIds1.union(movieIds2).distinct()

    //val movie = movieFeatureDF.filter(s"movieId==$movieId")
    //只有一行
    val movie = movieFeatureDF.where(s"movieId==$movieId").toDF()
    val allColumns = movie.columns
    var movieAlike :RDD[Any] = spark.sparkContext.emptyRDD

    //先找出与当前电影类型相同的所有电影吧
    import spark.implicits._
    for (title <- allColumns){
      if (1==movie.select(title).take(1){0}{0}){
        //不能把整个条件和列名都放到双引号中，因为，如果列名中有-，则会被当成两列的差值
        //val movie1 = movieFeatureDF.where(s"${title}==1").rdd.map(x=>x(0))
        import spark.implicits._

        //要像这样，把列名整个地放到一个字符串中，才能好使的
        val movie1 = movieFeatureDF.filter($"${title}" === 1).rdd.map(x=>x(0))
        movieAlike = movieAlike.union(movie1)
        //isComedy
        //val movie1 = movieFeatureDF.where(s"isComedy==1").toDF()
        //val movie1 = movieFeatureDF.where(s"isComedy==1").rdd.map(x=>x(0))

        //val movie2 = movieFeatureDF.filter($"isSci-Fi" === 1).toDF()
        ///val movie2 = movieFeatureDF.filter($"isSci-Fi" === 1).toDF()

      }
      //movie.select("isComedy").take(1){0}{0}
    }

    val movie_left_all = movieAlike.distinct().subtract(movieIds)
    //我要在这里头随机选20%吧
    //val movie_left = movie_left_all.sample(false,0.2)
    //不满100，就全部取出了哦~   另：此时的movie_left是个Array了呢。。。不再是RDD了
    val movie_left = movie_left_all.takeSample(false,100)

    //然后再放到rank模型中去排序就好了

    return movie_left

  }

  // userId movieId rating
  // test数据的rating可以不要，但是模型中已经有了，那就暂时写个0.0吧
  def constructUserMovie2DataFrame(spark: SparkSession, userId : Long, movieIdList : Array[Any]): DataFrame ={

    var retSeq = Seq[(Long, Long, Double)]()
    for (movie <- movieIdList){
      retSeq = retSeq ++ Seq((userId, movie.asInstanceOf[Long],0.0))
    }

    import spark.implicits._
    val ret = retSeq.toDF("userId","movieId","rating")
    ret
  }
}
