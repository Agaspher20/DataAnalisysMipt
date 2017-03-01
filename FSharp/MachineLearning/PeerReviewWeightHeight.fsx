#r @"..\packages\MathNet.Numerics.Signed.3.17.0\lib\net40\MathNet.Numerics.dll"
#r @"..\packages\MathNet.Numerics.FSharp.Signed.3.17.0\lib\net40\MathNet.Numerics.FSharp.dll"
#r @"..\packages\FSharp.Data.2.3.2\lib\net40\FSharp.Data.dll" 
#r @"..\packages\FSharp.Charting.0.90.14\lib\net40\FSharp.Charting.dll"
#r @"..\packages\Accord.3.4.0\lib\net46\Accord.dll"
#r @"..\packages\Accord.Math.3.4.0\lib\net46\Accord.Math.dll"
#r @"..\packages\Accord.Statistics.3.4.0\lib\net46\Accord.Statistics.dll"
#r @"bin\Debug\LibExtensions.dll"

open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Statistics
open FSharp.Charting
open FSharp.Data
open Accord
open Accord.Statistics.Models.Regression.Linear
open LibExtensions.Charting

[<Literal>]
let DataPath = __SOURCE_DIRECTORY__ + @"..\..\..\Data\weights_heights.csv"
type Data = CsvProvider<DataPath, Schema="int,float,float">
type Observation = Data.Row

let formatRow (row:Observation) =
    sprintf "%-7i %-7.3f %-7.3f" row.Index row.Height row.Weight

let dataSet = Data.Load(DataPath)
let data = dataSet.Rows
(match dataSet.Headers with
    | Some heads -> heads
    | None -> [||])
|> Array.map (sprintf "%-7s")
|> (String.concat " ")
|> (printfn "%s")
data |> Seq.take 5 |> Seq.iter (formatRow >>(printfn "%s"))

[
    data
    |> Seq.map(fun obs -> obs.Height)
    |> Chart.Histogram
    |> maxY 2700.
    |> namedAs "Height distribution"
    |> titleX "Height";
    data
    |> Seq.map(fun obs -> obs.Weight)
    |> Chart.Histogram
    |> maxY 2800.
    |> namedAs "Weight distribution"
    |> titleX "Weight"
]
|> Chart.Rows
|> namedAs "Height and Weight distribution"
|> Chart.Show

let makeBmi height weight =
    let meterToInch, kiloToPound = 39.37,2.20462
    (weight/kiloToPound)/(height/meterToInch)**2.
let dataBmi = data
              |> Seq.map(fun obs -> [obs.Height;obs.Weight;(makeBmi obs.Height obs.Weight)])
let titles = [|"Height";"Weight";"BMI"|]
let pairPlotData = [|
                    dataBmi |> Seq.map (fun l -> l.[0]);
                    dataBmi |> Seq.map (fun l -> l.[1]);
                    dataBmi |> Seq.map (fun l -> l.[2])
                   |]
(titles,pairPlotData)
||> pairPlot
|> namedAs "Pair plot"
|> Chart.Show

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
|> minY 64.
|> maxY 73.
|> Chart.Show

data
|> Seq.map(fun obs -> (obs.Weight, obs.Height))
|> Chart.Point
|> minY 59.
|> maxY 76.
|> titleX "Weight"
|> titleY "Height"
|> namedAs "Height from Weight dependency"
|> Chart.Show

let linearApproximation ((w0:float),(w1:float)) (weight:float) =
    w0 + w1*weight
let quadError w (data:seq<Observation>) =
    data
    |> Seq.sumBy (fun obs -> (obs.Height - (linearApproximation w obs.Weight))**2.)

[
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
|> Chart.Combine
|> minY 55.
|> maxY 80.
|> titleX "Weight"
|> titleY "Height"
|> Chart.Show

seq { -5. .. 0.2 .. 5. }
|> Seq.map(fun x -> (x, quadError (50., x) data))
|> Chart.Line
|> withTitle "Q(w2)"
|> namedAs "Quadratic error from w2 dependency"
|> titleX "w2"
|> titleY "Q"
|> Chart.Show

// didn't find method to optimize only coefficient
let ols = OrdinaryLeastSquares()
let regression = ols.Learn(
                     data |> Seq.map(fun obs -> obs.Weight) |> Seq.toArray,
                     data |> Seq.map(fun obs -> obs.Height) |> Seq.toArray
                     )
let w = (regression.Intercept,regression.Slope)
[
    data
    |> Seq.map(fun obs -> (obs.Weight, obs.Height))
    |> Chart.Point
    data
    |> Seq.map(fun obs -> obs.Weight,linearApproximation w obs.Weight)
    |> Chart.Line
]
|> Chart.Combine
|> minY 55.
|> maxY 80.
|> titleX "Weight"
|> titleY "Height"
|> Chart.Show
