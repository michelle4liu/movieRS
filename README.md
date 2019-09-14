#修改错别字，受不了错别字。
# movieRS
movie recommendation with spark

此项目开始于2019年4月初，结束于2019年6月初。  
作为一只萌新，从2018年11月开始学习DL和ML，学到2019年2月底的时候，基础知识和重难点都了解了，加上我自己还在天池做了一个小小的项目，我决定出去面试看看。整个3月份，传说中的金三银四，可是我拿到的面试机会却很少。。。仅有的几次面试机会，也面得不好。。。场面一度很尴尬哈哈哈  
我总结了下原因：首先学艺不精，对知识没有自己的理解与体会，在别人看来，就是在纯背诵。。。  
               其次，没有具体的方向。什么方向都觉得可能，什么都涉猎，没有focus on的方向，知识都很散。。。  
于是，我给自己定了一个确切的目标，那就是从文本推荐着手。其他知识点都沿着这条线来走。  
  
  
OK，正文开始~  
  
此推荐系统为近实时的个性化电影推荐系统，使用spark-streaming来对用户进行实时画像，从而根据实时画像更新推荐列表来达到近实时的效果。
推荐部分包括了召回和排序。召回部分只是简单地基于内容做了过滤，排序部分简单地使用了LR，其中包括用户特征和电影特征。  
  
后续有时间再完善。  
比如：召回部分，目前的规则是：用户u1对电影i1进行了评分，那么召回与电影i1类型相同的其他电影，同时过滤掉用户u1已经评分过的电影。  
               如果放到实际应用中，这个召回明显不行哈哈哈  
               完善的话：  
                     1、可以增加很多topList  
                     2、也可以用ALS(当时没有用是考虑到冷启动的问题)，但是增量训练可能是个问题，我看到有人提出了解决办法，可惜，我貌似木有看懂。。。  
                      3、协同过滤(增量训练是可以实现的)，有空的话实现一下，再测试一下性能  
      排序部分：目前的情况是：把用户(性别、年龄、职业)和电影特征(title、年代、类型)组合起来，线下训练一个LR，直接放到线上进行排序即可。  
                但是这个排序会存在冷启动的问题。如果用户是新用户，且不提供性别年龄和职业，怎么办？  
                                         如果电影是新电影，OK，这个不是问题，后台可以保证的是，在用户看见这个电影以前，把电影特征输入到系统中。  
                完善的话：1、解决用户冷启动问题  
                         2、模型欠拟合。。。可以尝试一下特征组合FM、神经网络Wide&Deep  
  
推荐中的难点：我感觉，最难的，是做实时推荐。  
             不仅召回要快，做实时画像也要快。。。模型还要增量跟新。。。  
               
                           
                           
                           
最后收获：  
        1、搭大数据运行环境、开发环境就搞了两周。。。  
           被spark-streaming kafka的依赖包冲突虐了好几天，我是有多废材【捂脸  
           本来想试试传说中强大的pom。。。可是。。。冲突冲突冲突。。。最后一个一个导依赖包才解决。。。  
        2、把以前一个旧的笔记本直接装成centos变成我的服务器啦哈哈哈，我爱linux，有linux才有安全感  
        3、系统学习了推荐系统，包括推荐系统框架的发展历史，重点学习目前比较流行的框架，又重新手推了SVD、ALS、FM，加深了记忆和理解。  
        4、回过头去巩固了LR和DNN，把以前懵懵懂懂的地方都搞清楚了。  
        5、巩固了spark中的RDD和DataFrame编程。算子不能嵌套！不能嵌套！不能嵌套！   
           以及，加深了对分布式并行处理的理解   
        6、终于抽时间看word2vec的训练过程了。。。  
          读了《word2vec Parameter Learning Explained》by Xin Rong，茅塞顿开。。。  
        6、要更加自律，提高效率~~~~~~~~~~~~~~~  
       
      
      
    
     
