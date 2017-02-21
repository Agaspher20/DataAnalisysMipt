#I @"..\packages\"
#r @"MathNet.Numerics.Signed.3.17.0\lib\net40\MathNet.Numerics.dll"
#r @"MathNet.Numerics.FSharp.Signed.3.17.0\lib\net40\MathNet.Numerics.FSharp.dll"
#r @"FSharp.Data.2.3.2\lib\net40\FSharp.Data.dll" 
#r @"Accord.3.4.0\lib\net46\Accord.dll"
#r @"Accord.Math.3.4.0\lib\net46\Accord.Math.dll"
#r @"Accord.Statistics.3.4.0\lib\net46\Accord.Statistics.dll"
#load @"FSharp.Charting.0.90.14\FSharp.Charting.fsx"

open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Statistics
open FSharp.Charting
open FSharp.Data
open Accord
open Accord.Statistics.Models.Regression.Linear

[<Literal>]
let DataPath = __SOURCE_DIRECTORY__ + @"..\..\..\Data\weights_heights.csv"
type Data = CsvProvider<DataPath, Schema="int,float,float">
type Observation = Data.Row

let dataSet = Data.Load(DataPath)
let data = dataSet.Rows

printfn "%A" dataSet.Headers
data |> Seq.take 5 |> Seq.iter (printfn "%A")

// Height distribution
data |> Seq.map(fun obs -> obs.Height) |> Chart.Histogram

// Weight distribution
data |> Seq.map(fun obs -> obs.Weight) |> Chart.Histogram

let makeBmi height weight =
    let meterToInch, kiloToPound = 39.37,2.20462
    (weight/kiloToPound)/(height/meterToInch)**2.

let dataBmi = data
              |> Seq.map(fun obs -> (obs.Height,obs.Weight,(makeBmi obs.Height obs.Weight)))
// didn't find pair plot
let makeWeightCategory weight =
    match weight with
    | w when w < 120. -> 1
    | w when w < 150. -> 2
    | _ -> 3
data
    |> Seq.map(fun obs -> (obs.Height,(makeWeightCategory obs.Weight)))
    |> Seq.groupBy(fun (_,w) -> w)
    |> Seq.map(fun(k,values) -> (k,values|>Seq.map fst))
    |> Chart.BoxPlotFromData

data
    |> Seq.map(fun obs -> (obs.Weight, obs.Height))
    |> Chart.Point

let linearApproximation ((w0:float),(w1:float)) (weight:float) =
    w0 + w1*weight
let quadError w (data:seq<Observation>) =
    data
    |> Seq.sumBy (fun obs -> (obs.Height - (linearApproximation w obs.Weight))**2.)

Chart.Combine [
    data
    |> Seq.map(fun obs -> (obs.Weight, obs.Height))
    |> Chart.Point
    seq { 60. .. 1. .. 180. }
    |> Seq.map (fun x -> (x, linearApproximation (60.,0.05) x))
    |> Chart.Line
    seq {60. .. 1. .. 180.}
    |> Seq.map (fun x -> (x, linearApproximation (50.,0.16) x))
    |> Chart.Line
]

seq { -5. .. 0.2 .. 5. }
|> Seq.map(fun x -> (x, quadError (50., x) data))
|> Chart.Line

// didn't find method to optimize only coefficient

let ols = OrdinaryLeastSquares()
let regression = ols.Learn(
                     data |> Seq.map(fun obs -> obs.Weight) |> Seq.toArray,
                     data |> Seq.map(fun obs -> obs.Height) |> Seq.toArray)
let w = (regression.Intercept,regression.Slope)
Chart.Combine [
    data
    |> Seq.map(fun obs -> (obs.Weight, obs.Height))
    |> Chart.Point
    data
    |> Seq.map(fun obs -> obs.Weight,linearApproximation w obs.Weight)
    |> Chart.Line
]
