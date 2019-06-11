import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, Word2Vec}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.{udf, when}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types._

object featureEngineering {
  val moviesFile="/movie/data/movies.dat"
  val moviesFeature="/movie/data/rank/movie/"
  val moviesFeature2="/movie/data/rank/movie2/"

  val userFile="/movie/data/users.dat"
  val userFeature="/movie/data/rank/user/"
  val userFeature2="/movie/data/rank/user2/"

  val userMovieRatings="/movie/data/ratings.dat"
  val ratingTrain="/movie/data/recall/rating/train/"
  val ratingTest="/movie/data/recall/rating/test/"

  def main(args : Array[String]): Unit ={

    val argLen= args.length
    if (argLen != 1){
      System.out.println()
      System.exit(1)
    }

    val spark=SparkSession.builder().getOrCreate()
    if (args{0}.contains("all")){
      //给电影数据做特征工程
      feature_engineering_movie(spark)

      //给用户数据做特征工程
      feature_engineering_user(spark)

      //给rating数据做特征工程
      feature_engineering_rating(spark)

    }else if (args{0}.contains("movie")){
      feature_engineering_movie(spark)
    }else if (args{0}.contains("user")){
      feature_engineering_user(spark)
    }else if (args{0}.contains("rating")){
      feature_engineering_rating(spark)
    }

    spark.stop()
  }

  val movieSchema = StructType(Array(
    StructField("movieId",LongType,true),
    StructField("title",StringType,true),
    StructField("genres",StringType,true)
  ))
  val userSchema = StructType(Array(
    StructField("userId",LongType,true),
    StructField("gender",StringType,true),
    StructField("age",IntegerType,true),
    StructField("occupation",IntegerType,true),
    StructField("zipCode",StringType,true)
  ))
  val ratingSchema = StructType(Array(
    StructField("userId",LongType,true),
    StructField("movieId",LongType,true),
    StructField("rating",FloatType,true),
    StructField("timestamp",LongType,true)
  ))

  def feature_engineering_rating(spark : SparkSession): Unit ={
    val allRatings = spark.read.format("csv")
      .option("header","false")
      .schema(ratingSchema)
      .load(userMovieRatings)

    //把评分数据拆分成train和test
    var Array(training, test) = allRatings.randomSplit(Array(0.7, 0.3), seed = 12345)
    //把train和test存起来
    training.write.format("parquet").option("header","true").save(ratingTrain)
    test.write.format("parquet").option("header","true").save(ratingTest)
  }

  def feature_engineering_user(spark: SparkSession): Unit ={
    val df_user = spark.read.format("csv")
      .option("header","false")
      .schema(userSchema)
      .load(userFile)

    //先把gender转成数值的，再用one-hot encoding
    val indexer = new StringIndexer().setInputCol("gender").setOutputCol("genderIndex")
    val indexed = indexer.fit(df_user).transform(df_user)

    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("genderIndex", "occupation"))
      .setOutputCols(Array("genderVec", "occupationVec"))

    val model = encoder.fit(indexed)
    val encoded = model.transform(indexed)

    val encoded2 = encoded.drop("gender")
    val encoded3 = encoded2.drop("occupation")
    //val encoded4 = encoded3.drop("genderIndex")
    //encoded4.write.format("parquet").option("header","true").save(userFeature2)

    encoded3.write.format("parquet").option("header","true").save(userFeature2)

  }
  def feature_engineering_movie(spark: SparkSession): Unit ={
    val df_movie = process_movies_pre(spark,moviesFile)
    process_movies_genres(spark,df_movie)
    process_movies_title(spark,moviesFeature+"*")
  }

  private def process_movies_title(spark: SparkSession, file : String): Unit ={
    val movie = spark.read.format("parquet").option("header","true").load(file)

    val title = movie.select("title2")

    //把title embedding成一个三维的向量
    val word2Vec = new Word2Vec().setInputCol("text").setOutputCol("result").setVectorSize(3).setMinCount(0)
    //需要把title中的每一行都拆分成单词
    import spark.implicits._
    val title3 = title.map(x=>x.getString(0).split(" ")).toDF("text")
    val model = word2Vec.fit(title3)

    val result = model.transform(title3)

    //这是一个向量
    val vector = result.select("result")

    val vector2 = vector.map{case Row(v: Vector) => (v(0), v(1), v(2))}.toDF("tv1", "v2", "tv3")

    //把title的embedding合并到原来的movie中
    //不能用crossJoin，crossJoin是笛卡儿积合并。。。
    //val df = movie.crossJoin(vector2)
    val df = combine2DataFrame(spark, movie,vector2)

    //删除title2这一列
    val movie2 = df.drop("title2")
    //保存在文件中
    movie2.write.format("parquet").option("header","true").save(moviesFeature2)
  }

  private def combine2DataFrame(spark: SparkSession, df1: DataFrame, df2 : DataFrame) : DataFrame={
    val combinedRow = df1.rdd.coalesce(1).zip(df2.rdd.coalesce(1)). map({
      //convert both dataframe to Seq and join them and return as a row
      case (df1Data, df2Data) => Row.fromSeq(df1Data.toSeq ++ df2Data.toSeq)
    })

    val combinedSchema =  StructType(df1.schema.fields ++ df2.schema.fields)

    val finalDF = spark.sqlContext.createDataFrame(combinedRow, combinedSchema)

    finalDF
  }
  private def process_movies_genres(mySpark : SparkSession,df_movie: DataFrame): Unit ={
    //val df_movie = mySpark.read.csv(file)
    //val df_ratings = mySpark.read.csv("/movie/data/ratings.csv")
    //df_movie.show(10)
    //df_movie.select("genres").take(10).collect()
    import mySpark.implicits._

    //取出genres中所有的电影类型
    val types = df_movie.select("genres")
      .map(r => {r.getString(0).split("\\|").toList})
      .collect.toList
    var list_genres=types(0)
    for (lone <- types){
      list_genres=(list_genres:::lone).distinct
    }
    //增加列，相当于独热向量编码
    var df_movie_new = df_movie
    for (lone <- list_genres){
      df_movie_new = df_movie_new.withColumn("is"+lone, when(df_movie("genres").contains(lone) ,1).otherwise(0))
    }
    //注意：有246部电影没有电影类型，为"is(no genres listed)"
    //df_movie_new.write.format("parquet").save("/movie/data/preprocess/movies")
    //可以写一个模型来预测电影类型，这就需要更多关于电影的数据（那份tag的数据好像可以用哦）
    //删掉原来的电影类型列
    df_movie_new = df_movie_new.drop("genres")
    //这个select并不会修改df_movie_new，如果等于的话，也只是返回一列
    //df_movie_new.select("is(no genres listed)").alias("is_no_genres_listed")

    //把这个特殊符号改一改
    df_movie_new=df_movie_new.withColumnRenamed("is(no genres listed)","is_no_genres_listed")
    //从title列中获取
    //获取出电影中的年份
    /*
    var years = df_movie.select("title")
      .map(r => {var s=r.getString(0);var length=s.length; strToInt(s.slice(length-5,length-1))})
      .collect.toList
    */
    val addcolyear=udf(add_col_year_code)
    df_movie_new=df_movie_new.withColumn("year",addcolyear(df_movie_new("title")))

    //把title中的年份去掉,变成title2,title列删掉
    val modcoltitle=udf(mod_col_title_code)
    df_movie_new = df_movie_new.withColumn("title2",modcoltitle(df_movie_new("title")))
    df_movie_new = df_movie_new.drop("title")
    df_movie_new.write.format("parquet").option("header","true").save(moviesFeature)

    //var dfnew = df1.join(df2,Seq("ID"),"left")
  }

  //原始电影文件预处理
  //正常情况下是这种格式
  //1,Toy Story (1995),Animation|Children's|Comedy
  // 因为，中间的title有时候会被分成两部分或者三部分
  //两部分的时候是title+year
  //三部分的时候是title1+title2+year
  private def process_movies_pre(spark : SparkSession,file : String) ={

    val sc = spark.sparkContext
    val movieRdd = sc.textFile(file)

    val movieRddPre = movieRdd.flatMap(line=>{
      val lineNew = line.split("\\,")
      val length = lineNew.length
      var ret : List[(Long,String,String)] = Nil
      var movieId : String = null
      var title : String = null
      var title1 : String = null
      var year : String = null
      var genres : String = null
      if (3 == length){
        movieId = lineNew(0)
        title = lineNew(1)
        genres = lineNew(2)

        ret = List((strToLong(movieId),title,genres))
      }else if (4 == length){
        movieId = lineNew(0)
        title = lineNew(1)
        year = lineNew(2)
        genres = lineNew(3)
        ret = List((strToLong(movieId),title+" "+year,genres))
      }else if (5 == length){
        movieId = lineNew(0)
        title = lineNew(1)
        title1 = lineNew(2)
        year = lineNew(3)
        genres = lineNew(4)
        ret = List((strToLong(movieId),title+" "+ title1 + " " + year,genres))
      }

      ret
    })

    val movieRddPre2 = movieRddPre.map(x=>Row(x._1,x._2,x._3))
    val df_movie = spark.createDataFrame(movieRddPre2,movieSchema)

    df_movie
  }
  def modelRankEngineering(spark: SparkSession): Unit ={

  }

  def strToInt(str: String): Int = {
    val regex = """([0-9]+)""".r
    val res = str match{
      case regex(num) => num
      case _ => "0"
    }
    val resInt = Integer.parseInt(res)
    resInt
  }

  def strToLong(str: String): Long = {
    val regex = """([0-9]+)""".r
    val res = str match{
      case regex(num) => num
      case _ => "0"
    }
    val resInt = res.toLong
    resInt
  }
  // 自定义add year 列的udf的函数
  val add_col_year_code = (arg: String) => {
    val length=arg.length;
    strToInt(arg.slice(length-5,length-1))
  }
  // 自定义add year 列的udf的函数
  val mod_col_title_code = (arg: String) => {
    val length=arg.length;
    arg.dropRight(6)
  }
}
