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

let formatRow (row:Observation) =
    sprintf "%-7i %-7.3f %-7.3f" row.Index row.Height row.Weight
let maxX max (chart:ChartTypes.GenericChart) =
    chart.WithXAxis(Max=max)
let maxY max (chart:ChartTypes.GenericChart) =
    chart.WithYAxis(Max=max)
let minX min (chart:ChartTypes.GenericChart) =
    chart.WithXAxis(Min=min)
let minY min (chart:ChartTypes.GenericChart) =
    chart.WithYAxis(Min=min)
let titleX title (chart:ChartTypes.GenericChart) =
    chart.WithXAxis(Title=title)
let titleY title (chart:ChartTypes.GenericChart) =
    chart.WithYAxis(Title=title)
let withoutXAxis (chart:ChartTypes.GenericChart) =
    chart.WithXAxis(Enabled=false)
let withoutYAxis (chart:ChartTypes.GenericChart) =
    chart.WithYAxis(Enabled=false)
let namedAs name (chart:ChartTypes.GenericChart) =
    chart.WithStyling(Name=name)
let withTitle title (chart:ChartTypes.GenericChart) = chart.WithTitle title
let hideXLabels (chart:ChartTypes.GenericChart) =
    chart.WithXAxis(LabelStyle=ChartTypes.LabelStyle(FontSize=0.01))
let hideYLabels (chart:ChartTypes.GenericChart) =
    chart.WithYAxis(LabelStyle=ChartTypes.LabelStyle(FontSize=0.01))

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
                   |> Array.zip titles
                   |> Array.Parallel.map(fun (t,d) -> (t,d,d|>Seq.max,d|>Seq.min))
pairPlotData
|> Array.Parallel.mapi (fun i (ti,di,maxi,mini) ->
    pairPlotData
    |> Array.Parallel.mapi (fun j (tj,dj,maxj,minj) ->
        let setBounds = maxX maxi >> maxY maxj >> minX mini >> minY minj
        let ch = match i=j with
                 | true -> dj |> Chart.Histogram
                 | false -> (di,dj)
                            ||> Seq.zip
                            |> Chart.Point
                            |> setBounds
        let titledCh = match j with
                       | 0 -> match i with
                              | 2 -> ch |> titleY ti |> titleX tj
                              | _ -> ch |> titleY ti |> withoutXAxis
                       | _ -> match i with
                              | 2 -> ch |> titleX tj |> withoutYAxis
                              | _ -> ch |> withoutXAxis |> withoutYAxis
        titledCh |> hideXLabels |> hideYLabels)
    |> Chart.Columns)
|> Chart.Rows
|> namedAs "Pair plot"

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

data
|> Seq.map(fun obs -> (obs.Weight, obs.Height))
|> Chart.Point
|> minY 59.
|> maxY 76.
|> titleX "Weight"
|> titleY "Height"
|> namedAs "Height from Weight dependency"

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

seq { -5. .. 0.2 .. 5. }
|> Seq.map(fun x -> (x, quadError (50., x) data))
|> Chart.Line
|> withTitle "Q(w2)"
|> namedAs "Quadratic error from w2 dependency"
|> titleX "w2"
|> titleY "Q"

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
