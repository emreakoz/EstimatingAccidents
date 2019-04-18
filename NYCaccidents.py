import findspark
findspark.init()

from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
import seaborn as sns
sns.set(font_scale=1.5)
import matplotlib.pyplot as plt

#Let's 
sc = SparkContext(appName = "traffic")
lines = sc.textFile('NYC_accidents_clean.csv')
lines = lines.filter(lambda x: len(x.split(',')) == 28)


yearAccident = lines.map(lambda x: (x.split(',')[0][-2:],1))
totYearAccident = yearAccident.reduceByKey(lambda x,y: x+y).collect()

monthAccident = lines.map(lambda x: (x.split(',')[0][:2],1))
totMonthAccident = monthAccident.reduceByKey(lambda x,y: x+y).collect()

hourAccident = lines.filter(lambda x: len(x.split(',')[1]) == 4 or len(x.split(',')[1]) == 5)
hourAccident = hourAccident.map(lambda x: (x.split(',')[1][0],1) if x.split(',')[1][1] == ':' else (x.split(',')[1][:2],1))
totHourAccident = hourAccident.reduceByKey(lambda x,y: x+y).collect()


motoristFatalities = lines.map(lambda x: int(x.split(',')[16]))
totmotoristFatalities = motoristFatalities.reduce(lambda x,y: x+y)

cyclistFatalities = lines.map(lambda x: int(x.split(',')[14]))
totcyclistFatalities = cyclistFatalities.reduce(lambda x,y: x+y)

pedFatalities = lines.map(lambda x: int(x.split(',')[12]))
totpedFatalities = pedFatalities.reduce(lambda x,y: x+y)

def mapReducer(lines, colNum):
    mapper = lines.map(lambda x: ((x.split(',')[0][-2:],x.split(',')[colNum]),1))
    reducer = mapper.reduceByKey(lambda x,y: x+y).collect()
    return mapper, reducer

reasonv1, totReasonv1 = mapReducer(lines, 17)
reasonv2, totReasonv2 = mapReducer(lines, 18)
reasonv3, totReasonv3 = mapReducer(lines, 19)
reasonv4, totReasonv4 = mapReducer(lines, 20)
reasonv5, totReasonv5 = mapReducer(lines, 21)

reason = reasonv1.union(reasonv2)
reason = reason.union(reasonv3)
reason = reason.union(reasonv4)
reason = reason.union(reasonv5)
totReason = reason.reduceByKey(lambda x,y: x+y).collect()
sortedByFrequencyOfReason = sorted(totReason, key=lambda x: x[1],reverse=True)

carType1, totCarType1 = mapReducer(lines, 23)
carType2, totCarType2 = mapReducer(lines, 14)
carType3, totCarType3 = mapReducer(lines, 15)
carType4, totCarType4 = mapReducer(lines, 26)
carType5, totCarType5 = mapReducer(lines, 27)


carType = carType1.union(carType2)
carType = carType.union(carType3)
carType = carType.union(carType4)
carType = carType.union(carType5)
totCarType = carType.reduceByKey(lambda x,y: x+y).collect()
sortedByFrequencyOfCarType = sorted(totCarType, key=lambda x: x[1],reverse=True)


#lines = lines.filter(lambda x: len(x.split(',')[4]) > 0 and len(x.split(',')[5]) > 0)
#lines = lines.filter(lambda x: float(x.split(',')[4].encode('utf8')) > 35 and float(x.split(',')[4].encode('utf8')) < 45)
#lines = lines.filter(lambda x: float(x.split(',')[5].encode('utf8')) > -80 and float(x.split(',')[5].encode('utf8')) < -70)
#location = lines.map(lambda x: [float(x.split(',')[4].encode('utf8')),float(x.split(',')[5].encode('utf8'))])
#
#def error(point):
#    center = clusters.centers[clusters.predict(point)]
#    return  (sum([x**2 for x in (np.asarray(point) - np.asarray(center))]))**0.5
#
#error_list=[]
#for k in [5, 10, 15, 20, 25, 30, 35]:
#    clusters = KMeans.train(location, k, maxIterations=10,runs=10, initializationMode="random")
#    WSSSE = location.map(lambda point: error(point)).reduce(lambda x, y: x + y)
#    error_list.append(WSSSE)
#
#np.savetxt("./k_variance.csv", error_list, delimiter=",")
#print(error_list)        
#
#k_optimal = 20
#clusters = KMeans.train(location, k_optimal, maxIterations=30,runs=30, initializationMode="random")
#
#centersOfAccidents = []
#for i in range(0,k_optimal):
#    centeroid = clusters.centers[i]
#    centersOfAccidents.append(centeroid)
#
#np.savetxt("./centroids.csv", centersOfAccidents, delimiter=",")

###----------------------------------------------------------------------------------------------------###
#lines = lines.filter(lambda x: len(x.split(',')[4]) > 0 and len(x.split(',')[5]) > 0)
#lines = lines.filter(lambda x: float(x.split(',')[4].encode('utf8')) > 35 and float(x.split(',')[4].encode('utf8')) < 45)
#lines = lines.filter(lambda x: float(x.split(',')[5].encode('utf8')) > -80 and float(x.split(',')[5].encode('utf8')) < -70)
#
#linesAcc = lines.map(lambda x: (x.split(',')[0],(x.split(',')[10],x.split(',')[1],x.split(',')[4],x.split(',')[5])))
#
#linesW = sc.textFile('weather.csv')
#linesW = linesW.map(lambda x: (x.split(',')[2],x.split(',')[1]))
#lines = linesW.join(linesAcc)
#
#
#
#def parseData(line):
#    month = int(line[0][0:2])/12.0
#    lat = float(line[1][1][2]) - 40
#    lng = float(line[1][1][3]) + 75
#    weather = float(line[1][0])
#    
#    if len(line[1][1][1][0:2])==5:
#        time = int(line[1][1][1][0:2])/24.0
#    else:
#        time = int(line[1][1][1][0:1])/24.0
#    
#    if int(line[1][1][0])>0:
#        label = 1
#    else:
#        label = 0
#    
#    features = [weather, lat, lng, time, month]
#
#    return LabeledPoint(label,features)
#
#parsedData = lines.map(parseData)
#
#training , test = parsedData.randomSplit([ 0.7 , 0.3] , seed = 11L)
#model = LogisticRegressionWithLBFGS.train(training)
#
## Evaluating the model on training data
#labelsAndPreds = training.map(lambda p: (p.label, model.predict(p.features)))
#trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(training.count())
#print("Training Error = " + str(trainErr))


sc.stop()