#I @"..\packages\"
#r @"MathNet.Numerics.Signed.3.17.0\lib\net40\MathNet.Numerics.dll"
#r @"MathNet.Numerics.FSharp.Signed.3.17.0\lib\net40\MathNet.Numerics.FSharp.dll"
#load @"FSharp.Charting.0.90.14\FSharp.Charting.fsx"

open System
open MathNet.Numerics.Distributions
open MathNet.Numerics.Statistics
open FSharp.Charting

let calcMean (data:seq<float>) = MathNet.Numerics.Statistics.Statistics.Mean(data)
let line (min:float) (max:float) (data:seq<float*float>) = Chart.Line(data).WithXAxis(Min=min,Max=max)
let histogram (min:float) (max:float) (data:seq<float>) = Chart.Histogram(data, LowerBound = min, UpperBound=max)
let sqrt x = System.Math.Sqrt x

let min = 0.
let max = 1.
let arcSinDistrib = MathNet.Numerics.Distributions.Beta(0.5, 0.5)
let hist (data:seq<float>) = histogram min max data
let lin (data:seq<float*float>) = line min max data
    
let samples = arcSinDistrib.Samples() |> Seq.take 1000
let pdf = seq{ 0.001 .. 0.001 .. 0.999 } |> Seq.map(fun n -> n,arcSinDistrib.Density(n))
Chart.Rows([hist samples; lin pdf])

let buildRange (distribution:IContinuousDistribution) (size:int) (rangeSize:int) =
    seq { 0 .. rangeSize }
    |> Seq.map (fun i -> distribution.Samples() |> Seq.take size |> calcMean)

let buildNormalDistribution (mean:float) (variance:float) (min:float) (max:float) (size:int) (step:float) =
    let n = MathNet.Numerics.Distributions.Normal(mean, System.Math.Sqrt(variance/(float size)))
    seq { min .. step .. max }
    |> Seq.map (fun x -> x,n.Density(x))

let variance = 1./8. // variance
let mean = 0.5 // mean
let assembleData (size:int) =
    buildRange arcSinDistrib size 1000,buildNormalDistribution mean variance 0. 1. size 0.001

let fiveRange,fiveNormal = assembleData 5
Chart.Rows([ hist fiveRange; lin fiveNormal ])

let tenRange,tenNormal = assembleData 10
Chart.Rows([ hist tenRange; lin tenNormal ])

let fiftyRange,fiftyNormal = assembleData 50
Chart.Rows([ hist fiftyRange; lin fiftyNormal ])
