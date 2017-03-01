#r @"..\packages\MathNet.Numerics.Signed.3.17.0\lib\net40\MathNet.Numerics.dll"
#r @"..\packages\MathNet.Numerics.FSharp.Signed.3.17.0\lib\net40\MathNet.Numerics.FSharp.dll"
#r @"..\packages\FSharp.Data.2.3.2\lib\net40\FSharp.Data.dll"
#r @"bin\Debug\LibExtensions.dll"


open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra

open LibExtensions.Charting
open FSharp.Data

[<Literal>]
let DataPath = __SOURCE_DIRECTORY__ + @"..\..\..\Data\bikes_rent.csv"
type Data = CsvProvider<DataPath, Schema="int,int,int,int,int,int,int,float,float,float,float,float,int">
type Observation = Data.Row

let dataSet = Data.Load(DataPath)
let data = dataSet.Rows
let obsToList (obs:Observation) =
    [
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
    ]

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

let chartData = data |> Seq.map (fun obs -> (obs |> obsToList))
headers.[0..headers.Length-2]
    |> Array.mapi (fun i h ->
        let cd = chartData |> Seq.map (fun rl ->rl.[i],rl|>List.last)
        i/2,(h,cd))
    |> Array.groupBy (fun (k,_) -> k)
    |> Array.map(fun (_,c) ->
        let keys,charts = c |> Array.unzip
        Chart.Rows(
            charts
            |> Seq.map(fun (h,c) ->
                Chart.Point(c,
                            Name=sprintf "Count from %s dependency" h,
                            Title=sprintf "Count(%s)" h,
                            XTitle=h,
                            YTitle="Count",
                            Color=Drawing.Color.Red,
                            MarkerColor=Drawing.Color.Green))))