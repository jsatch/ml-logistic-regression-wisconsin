from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def init():
    conf = SparkConf().setAppName("WisconsinCancer").setMaster("local")
    return SparkContext(conf=conf)

def parseLinea(linea):
    sample = linea.split(",")
    if sample[6] != '?':
        return list(map(lambda x : int(x), sample))

def load_dataset(spark):
    rdd = spark.sparkContext.textFile(
        "data/wbcd.csv").map(lambda linea: linea.split(","))
    
    rdd_data = rdd.map(lambda sample: [
        sample[0], int(sample[1]), int(sample[2]),
        int(sample[3]), int(sample[4]), int(sample[5]),
        int(sample[6]) if sample[6] != '?' else 0, int(sample[7]), int(sample[8]),
        int(sample[9]), 
        0 if sample[10] == '2' else 1])
    headers = ["CODIGO", "CLUMP", "SIZE", "SHAPE",
        "ADH", "EPIT", "BNU", "BCHR", "NNU", "MIT",
        "CLASS"]
    return spark.createDataFrame(rdd_data, headers)

def prepare_dataset(data):
    train, test = data.randomSplit(
        [0.7, 0.3], seed=12345
    )
    train.show()
    headers_feature = ["CLUMP", "SIZE", "SHAPE",
        "ADH", "EPIT", "BNU", "BCHR", "NNU", "MIT"]
    header_output = "features"

    assembler = VectorAssembler(
        inputCols=headers_feature,
        outputCol=header_output)
    train_data = assembler.transform(train).select("features", "CLASS")
    test_data = assembler.transform(test).select("features", "CLASS")

    return train_data,test_data

def main():
    sc = init()
    spark = SparkSession(sc)
    data = load_dataset(spark)

    train_data, test_data = prepare_dataset(data)
    #train_data.show()
    #test_data.show()

    print("Encontrando h ....")

    lr = LogisticRegression(
        maxIter=100, regParam=0.3, elasticNetParam=0.8,
        labelCol='CLASS', family='binomial')
    
    lr_model = lr.fit(train_data)

    print("Coeficientes: " + str(lr_model.coefficients))
    print("Intercept: " + str(lr_model.intercept))

    print("Testing model...")

    data_to_validate = lr_model.transform(test_data)
    
    evaluator1 = BinaryClassificationEvaluator(
        labelCol='CLASS', metricName='areaUnderROC', 
        rawPredictionCol='rawPrediction'
    )
    print("{}:{}".format(
        "areaUnderROC",evaluator1.evaluate(data_to_validate)))

    evaluator2 = BinaryClassificationEvaluator(
        labelCol='CLASS', metricName='areaUnderPR', 
        rawPredictionCol='rawPrediction'
    )
    print("{}:{}".format(
        "areaUnderPR",evaluator2.evaluate(data_to_validate)))

if __name__ == '__main__':
    main()