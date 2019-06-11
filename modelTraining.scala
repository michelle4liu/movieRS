import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

object modelTraining {

  val ratingTrain="/movie/data/recall/rating/train/*"
  val ratingTest="/movie/data/recall/rating/test/*"
  val modelRecall="/movie/model/recall/"

  val userFeature2="/movie/data/rank/user2/"
  val moviesFeature2="/movie/data/rank/movie2/"
  val modelRank="/movie/model/rank/"

  def main(args:Array[String]): Unit ={

    val argLen = args.length
    if (argLen < 2){
      System.out.println("usage: spark2-submit ***.jar train/use rank/recall")
      System.exit(1)
    }
    val spark = SparkSession.builder().getOrCreate()

    val arg1 = args{0}
    val arg2 = args{1}

    if (arg1.contains("train")){
      if (arg2.contains("rank")){
        modelTrainingRank(spark)
      }else if (arg2.contains("recall")){
        modelTraingRecall(spark)
      }

    }else if (arg2.contains("use")){
      if (arg2.contains("rank")){
        //需要其他参数
        //modelUseRank(spark)
      }else if (arg2.contains("recall")){
        val modelRecall = modelLoadRecall(spark)
        modelUseRecall(spark,modelRecall)
      }
    }
    spark.stop()

  }

  private def modelTrainingRank(spark: SparkSession): Unit ={
    val df_users = spark.read.format("parquet").option("header","true").load(userFeature2)
    val df_movie = spark.read.format("parquet").option("header","true").load(moviesFeature2)

    val df_train_rating = spark.read.format("parquet").option("header","true").load(ratingTrain)

    //raing先和user特征结合
    val users_2 = df_train_rating.select("userId","movieId","rating")
    val users_3 = users_2.join(df_users,df_users("userId")===users_2("userId"),"left").drop(df_users("userId"))
    //再和movie特征结合
    val movie = users_3.join(df_movie,users_3("movieId")===df_movie("movieId"),"left").drop(df_movie("movieId"))
    val movie2 = movie.drop("zipCode","genderIndex")

    val assembler = new VectorAssembler().setInputCols(movie2.columns).setOutputCol("my")
    val output = assembler.transform(movie2)
    //sparseVector的格式
    //(1,[],[])
    // (20,[10],[1.0])
    // (47,[0,1,2,3,15,2...
    //不需要把userId和movieId也放到feature里面
    val k = output.select("my").rdd.map(x=>{
      //首先取出第0列，第0列是DenseVctor类型
      val myVector = x(0).asInstanceOf[SparseVector].toDense
      val length = myVector.size
      var ret : String = null
      var value : Double = 0
      for (i <- 0 until length){
        value = myVector(i)
        if (i == 2){
          ret = value.toString
        }
        if(i > 2){
          if (0 != value){
            ret = ret + " " + (i-2).toString + ":" + value
          }
        }

      }
      ret
    })
    k.saveAsTextFile("/libsvm")

    //然后可以开始用LR
    //读出来是两列，[label: double, features: vector]
    val training = spark.read.format("libsvm").load("/libsvm/*")
    //val lr = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0)
    //α(λ∥w∥1)+(1−α)(λ2∥w∥22),α∈[0,1],λ≥0
    //elasticNetParam corresponds to α and regParam corresponds to λ.
    val lr = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0)
    val lrModel = lr.fit(training)

    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

    lrModel.save(modelRank)
  }

  private def modelLoadRank(spark: SparkSession): LinearRegressionModel ={
    val LrModel = LinearRegressionModel.load(modelRank)
    LrModel
  }

  private def modelUseRank(spark : SparkSession,
                           lrModel : LinearRegressionModel,
                           testRatingDF : DataFrame,
                           userFeatureDF : DataFrame,
                           movieFeatureDF : DataFrame): DataFrame ={

    val testRatingDF2 = testRatingDF.select("userId","movieId","rating")

    //从userFeatureDF中去寻找用户的特征
    // 如果用户特征中没有这个用户的特征，那么对应的特征列中的值就为null，
    // 如果为null，放到模型中会怎么样呢？这个后续再测试
    val testRatingUser = testRatingDF2.join(userFeatureDF,userFeatureDF("userId")===testRatingDF2("userId"),"left").drop(userFeatureDF("userId"))
    val testRatingUser2 = testRatingUser.drop("zipCode","genderIndex")

    //再和movie特征结合
    val testratingUserMovie = testRatingUser2.join(movieFeatureDF,movieFeatureDF("movieId")===testRatingUser2("movieId"),"left").drop(movieFeatureDF("movieId"))
    //这个地方的movieID应该不会是空的，可以在流程上进行处理，以得到保证

    import java.util.Arrays

    //特征列是 "userId" "movieId" "rating"
    // 所以从第3列开始，即 age特征开始
    val assembler_columns = Arrays.copyOfRange(testratingUserMovie.columns,3,testratingUserMovie.columns.length)
    val t_assembler = new VectorAssembler().setInputCols(assembler_columns).setOutputCol("features")
    val t_output = t_assembler.transform(testratingUserMovie)

    // 传入一个dataSet，会自动去找“features”列, features列是一个Sparse Vector
    // 然后生成一个"prediction"列
    //预测完后还是一个DataFrame
    val prediction = lrModel.transform(t_output.select("userId","movieId","rating","features"))

    prediction
  }

  //这个模型是ALS，但是，需要注意的是，使用的时候，一定要考虑到冷启动的问题
  private def modelTraingRecall(spark:SparkSession){
    val train = spark.read.format("parquet").option("header","true").load(ratingTrain)
    val test = spark.read.format("parquet").option("header","true").load(ratingTest)

    //rank是隐向量的维度
    val als = new ALS().setRank(50).setMaxIter(5).setRegParam(0.01).setUserCol("userId").setItemCol("movieId").setRatingCol("rating");
    val model = als.fit(train)

    model.setColdStartStrategy("drop")
    val predictions = model.transform(test)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)

    model.save(modelRecall+rmse+"/")
  }

  private def modelLoadRecall(spark:SparkSession): ALSModel ={
    val model = ALSModel.load(modelRecall)

    model
  }

  private  def modelUseRecall(spark:SparkSession, model:ALSModel): Unit ={
    val test = spark.read.format("parquet").option("header","true").load(ratingTest)

    val users = test.select("userId").distinct().limit(3)
    //返回的是前100个
    //rec: org.apache.spark.sql.DataFrame =
    // [userId: int, recommendations: array<struct<movieId:int,rating:float>>]
    val rec = model.recommendForUserSubset(users,100)
    //先召回1百个，再对这1百个进行排序
    for (one <- rec) {
      val userId = one(0);
      val WrappedArray = one(1);

      /*
      for (movie <- WrappedArray){
        var movidId = movie(0)
        //这里就可以产生一个样本了。
        //user movie
        //组合成rank模块需要的数据类型
        //放入rank模型即可
      }
      */
    }
  }


}
