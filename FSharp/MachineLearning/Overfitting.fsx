#r @"..\packages\MathNet.Numerics.Signed.3.17.0\lib\net40\MathNet.Numerics.dll"
#r @"..\packages\MathNet.Numerics.FSharp.Signed.3.17.0\lib\net40\MathNet.Numerics.FSharp.dll"
#r @"..\packages\FSharp.Data.2.3.2\lib\net40\FSharp.Data.dll"
#r @"..\packages\FSharp.Charting.0.90.14\lib\net40\FSharp.Charting.dll"
#r @"bin\Debug\LibExtensions.dll"


open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra

open FSharp.Data
open FSharp.Charting

open LibExtensions.Charting

[<Literal>]
let DataPath = __SOURCE_DIRECTORY__ + @"..\..\..\Data\bikes_rent.csv"
type Data = CsvProvider<DataPath, Schema="int,int,int,int,int,int,int,float,float,float,float,float,int">
type Observation = Data.Row

let dataSet = Data.Load(DataPath)
let data = dataSet.Rows
let obsToArray (obs:Observation) =
    [|
        obs.Season |> float;
        obs.Yr |> float;
        obs.Mnth |> float;
        obs.Holiday |> float;
        obs.Weekday |> float;
        obs.Workingday |> float;
        obs.Weathersit |> float;
        obs.Temp;
        obs.Atemp;
        obs.Hum;
        obs.``Windspeed(mph)``;
        obs.``Windspeed(ms)``;
        obs.Cnt |> float
    |]

let formatObservation (obs:Observation) =
    sprintf "%6i %2i %4i %7i %7i %10i %10i %4.1f %5.2f %3.2f %12f %13f %3i"
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
let pairPlotData = data |> Seq.map obsToArray |> Seq.toList

headers.[0..headers.Length-2]
|> Array.mapi (fun i h -> i,h,(pairPlotData|>Seq.map(fun a -> a.[i])))
|> Array.groupBy (fun (idx,_,_) -> idx/4)
|> Array.map (fun (_,columns) ->
    columns
    |> Array.mapi (fun idx (i,title,column) ->
        let ch = (column,counts)
                 ||> Seq.zip
                 |> Chart.Point
                 |> titleX title
                 |> hideXLabels
                 |> hideYLabels
        match idx with
        | 0 -> ch |> titleY "Cnt"
        | _ -> ch
    )
    |> Chart.Columns)
|> Chart.Rows
|> Chart.Show
