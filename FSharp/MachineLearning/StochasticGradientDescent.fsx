#I @"..\packages\"
#r @"FSharp.Data.2.3.2\lib\net40\FSharp.Data.dll" 
#r @"MathNet.Numerics.Signed.3.17.0\lib\net40\MathNet.Numerics.dll"
#r @"MathNet.Numerics.FSharp.Signed.3.17.0\lib\net40\MathNet.Numerics.FSharp.dll"
#load @"FSharp.Charting.0.90.14\FSharp.Charting.fsx"

open System
open System.IO
open FSharp.Data
open MathNet.Numerics
open MathNet.Numerics.Statistics
open MathNet.Numerics.LinearAlgebra
open FSharp.Charting

[<Literal>]
let resultPath = __SOURCE_DIRECTORY__ + @"..\..\..\Results\"
[<Literal>]
let DataPath = __SOURCE_DIRECTORY__ + @"..\..\..\Data\advertising.csv"

let writeToFile path name (result:float) =
    use file = File.CreateText(path + name)
    file.Write(Math.Round(result))
    file.Close()

let writeResult = writeToFile resultPath

type Data = CsvProvider<DataPath, Schema="int,float,float,float,float">
type Observation = Data.Row

let dataSet = Data.Load(DataPath)

dataSet.Headers.Value
|> String.concat "\t"
|> (printfn "%s")
dataSet.Rows
|> Seq.take 5
|> Seq.iter (printfn "%A")

let X,y = dataSet.Rows
          |> Seq.map (fun row -> (row.TV,row.Radio,row.Newspaper),row.Sales)
          |> Seq.toList
          |> List.unzip

let means,stds = X
                 |> List.unzip3
                 |> (fun (tv,radio,newspaper) ->
                     (Statistics.Mean(tv)
                     ,Statistics.Mean(radio)
                     ,Statistics.Mean(newspaper))
                     ,
                     (Statistics.StandardDeviation(tv)
                     ,Statistics.StandardDeviation(radio)
                     ,Statistics.StandardDeviation(newspaper)))
let scale (x:float) (mean:float) (std:float) = (x - mean)/std
let scaleRow (tv,radio,newspaper)
             (tvm, radiom, newspaperm)
             (tvs, radios, newspapers) =
    (scale tv tvm tvs
    ,scale radio radiom radios
    , scale newspaper newspaperm newspapers)
let scaledX = X
              |> List.map(fun obs -> scaleRow obs means stds)
              |> List.map(fun (tv,radio,news) -> [tv;radio;news;1.])

let msError (ys:List<float>) (yPredicteds:List<float>) =
    let length = ys |> List.length |> float
    let sum = (ys, yPredicteds)
                ||> List.zip
                |> List.sumBy(fun (y, yPred) -> (y-yPred)**2.)
    sum/length

let median = Statistics.Median(y)
let answer1 = List.init (y.Length) (fun index -> median) |> msError y
writeResult "1.txt" answer1
printfn "%.3f" answer1

let matrixX = scaledX |> matrix
let vectorY = vector y

let normalEquation (X:Matrix<float>) (y:Vector<float>) =
    let Xt = X.Transpose()
    Xt.Multiply(X).Solve(Xt.Multiply(y))

let normalEquationWeights = normalEquation matrixX vectorY |> Seq.toList
printfn "%A" normalEquationWeights

let linearModel (w:List<float>) (x:List<float>) =
    (x,w)
    ||> List.zip
    |> List.sumBy(fun (xi,wi) -> xi*wi)

let answer2 = [0.;0.;0.;1.] |> linearModel normalEquationWeights

writeResult "2.txt" answer2
printfn "%.3f" answer2

let linearPrediction w X = X |> List.map (linearModel w)

let answer3 = scaledX |> (linearPrediction normalEquationWeights) |> (msError y)

writeResult "3.txt" answer3
printfn "%.3f" answer3

let updateWeights eta
                  (length:int)
                  (yk:float)
                  (xk:List<float>)
                  (w:List<float>) =
    let l = length |> float
    let stepDiff = ((w,xk)
                    ||> List.zip
                    |> List.sumBy (fun (wv,x) -> wv*x)) - yk
    let diff = xk
                |> List.map (fun x -> x * 2.*eta/l*stepDiff)
    (w,diff) ||> List.zip |> List.map (fun (wv, d) -> wv-d)

let stochasticGradientDescentStep (rnd:Random)
                           eta
                           (y:List<float>)
                           (X:List<List<float>>)
                           prevWeights =
    let randomIndex = rnd.Next(0, y.Length)
    let weights = (y.[randomIndex], X.[randomIndex], prevWeights)
                    |||> updateWeights eta y.Length
    let error = (weights,X)
                ||> linearPrediction
                |> (msError y)
    weights,error

let stochasticGradientDescent seed
                              eta
                              minWeightDist
                              maxIter
                              y
                              X
                              wInit =
    let mutable iterNum = 0
    let mutable distance = Double.MaxValue
    let mutable weights = wInit
    let mutable errors = []
    let rnd = Random(seed)
    while (distance > minWeightDist) && iterNum < maxIter do
        let (stepWeights,stepError) = weights |> stochasticGradientDescentStep rnd eta y X
        distance <- (stepWeights,weights)
            ||> List.zip
            |> List.sumBy(fun (w,pw) -> (w-pw)**2.)
            |> (fun v -> Math.Sqrt(v))
        weights <- stepWeights
        errors <- [stepError] |> List.append errors
        iterNum <- iterNum + 1
    (weights,errors)

let gradientWeights,gradientErrors = [0.;0.;0.;0.] |> stochasticGradientDescent 42 0.01 (1./10.**8.) 100000 y scaledX

gradientErrors |> Chart.Line

let answer4 = gradientErrors |> List.last

writeResult "4.txt" answer4
printfn "%.3f" answer4