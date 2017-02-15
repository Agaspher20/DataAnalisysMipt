#I @"..\packages\"
#r @"MathNet.Numerics.Signed.3.17.0\lib\net40\MathNet.Numerics.dll"
#r @"MathNet.Numerics.FSharp.Signed.3.17.0\lib\net40\MathNet.Numerics.FSharp.dll"
#load @"FSharp.Charting.0.90.14\FSharp.Charting.fsx"

open MathNet.Numerics.Distributions
open FSharp.Charting

let line (data:seq<float*float>) = Chart.Line(data)
let histogram (data:seq<float>) = Chart.Histogram(data, UpperBound=2.5, LowerBound = 0.)

let distribution = MathNet.Numerics.Distributions.Beta(2., 5.)
let samples = distribution.Samples() |> Seq.take 1000
let pdf = seq{ 0. .. 0.01 .. 1. } |> Seq.map(fun n -> n,distribution.Density(n))

//Chart.Combine([
//                histogram samples;
//                line pdf
//              ])
histogram samples
line pdf
