#r @"..\packages\Accord.3.4.0\lib\net46\Accord.dll"
#r @"..\packages\Accord.Math.3.4.0\lib\net46\Accord.Math.dll"
#r @"..\packages\Accord.Statistics.3.4.0\lib\net46\Accord.Statistics.dll"
#r @"..\packages\FSharp.Data.2.3.2\lib\net40\FSharp.Data.dll"
#r @"..\packages\FSharp.Charting.0.90.14\lib\net40\FSharp.Charting.dll"
#r @"bin\Debug\LibExtensions.dll"


open System
open Accord
open Accord.Math
open Accord.Statistics
open Accord.Statistics.Models.Regression.Linear

open FSharp.Data
open FSharp.Charting

open LibExtensions.Charting

[<Literal>]
let DataPath = __SOURCE_DIRECTORY__ + @"..\..\..\Data\bikes_rent.csv"
type Data = CsvProvider<DataPath, Schema="float,float,float,float,float,float,float,float,float,float,float,float,float">
type Observation = Data.Row

let dataSet = Data.Load(DataPath)
let data = dataSet.Rows
let obsToArray (obs:Observation) =
    [|
        obs.Season;
        obs.Yr;
        obs.Mnth;
        obs.Holiday;
        obs.Weekday;
        obs.Workingday;
        obs.Weathersit;
        obs.Temp;
        obs.Atemp;
        obs.Hum;
        obs.``Windspeed(mph)``;
        obs.``Windspeed(ms)``;
        obs.Cnt
    |]

let formatObservation (obs:Observation) =
    sprintf "%6.0f %2.0f %4.0f %7.0f %7.0f %10.0f %10.0f %4.1f %5.2f %3.2f %12f %13f %3.0f"
            obs.Season
            obs.Yr
            obs.Mnth
            obs.Holiday
            obs.Weekday
            obs.Workingday
            obs.Weathersit
            obs.Temp
            obs.Atemp
            obs.Hum
            obs.``Windspeed(mph)``
            obs.``Windspeed(ms)``
            obs.Cnt
let headers = match dataSet.Headers with
              | Some h -> h
              | None -> [||]

String.concat " " (headers |> Array.toSeq)  |> printfn "%s"
data
|> Seq.take 5
|> Seq.iter (formatObservation >> (printfn "%s"))

let counts = data |> Seq.map(fun obs -> obs.Cnt |> float) |> Seq.toList
let dataMatrix = data |> Seq.map obsToArray |> Seq.toArray
let transposedData = Matrix.Transpose<float>(dataMatrix)
(headers.[..headers.Length-2],transposedData.[..headers.Length-2])
||> Seq.zip
|> Seq.mapi (fun i d -> i,d)
|> Seq.groupBy (fun (idx,d) -> idx/4)
|> Seq.map (fun (_,columns) ->
    columns
    |> Seq.mapi (fun idx (i,(title,column)) ->
        let ch = (column,counts)
                 ||> Seq.zip
                 |> Chart.Point
                 |> titleX title
                 |> xLabelsFontSize 6.
        match idx with
        | 0 -> ch |> titleY "Cnt" |> yLabelsFontSize 6.
        | _ -> ch |> yLabelsFontSize 0.1
    )
    |> Chart.Columns)
|> Chart.Rows
|> Chart.Show

Measures.Correlation(dataMatrix).[..headers.Length-2]
|> Seq.map(fun cols -> cols |> Seq.last)
|> Seq.zip headers.[..headers.Length-2]
|> Seq.iter(fun(h,c) -> printfn "%15s: %f" h  c)

([""], headers.[7..])
||> Seq.append
|> Seq.map (sprintf "%15s")
|> String.concat " "
|> printfn "%s"
Measures.Correlation(dataMatrix).[7..]
|> Seq.map(fun(cols) -> cols.[7..])
|> Seq.zip headers.[7..]
|> Seq.iter(fun (h,c) ->
    c
    |> Seq.map (sprintf "%15f")
    |> String.concat " "
    |> printfn "%15s %s" h)

transposedData
|> Seq.map(fun col -> Measures.Mean(col))
|> Seq.zip headers
|> Seq.iter(fun(h,c) -> printfn "%15s: %f" h c)

let X = 
    transposedData.[..headers.Length-2]
    |> Array.map (fun col ->
        Vector.Scale(col, DoubleRange(col|>Seq.min, col|>Seq.max), DoubleRange(0., 1.)))
    |> fun m -> Matrix.Transpose<float>(m)
    |> fun m -> Vector.Shuffled<float[]>(m)
let y = transposedData.[headers.Length-2]

let ols = OrdinaryLeastSquares()
let regression = ols.Learn(X, y)

(headers,regression.Weights)
||> Seq.zip
|> Seq.iter(fun(h,c) -> printfn "%15s: %f" h  c)

// Didn't find l1 and l2 regularization for linear regression algorithms
