#I @"..\packages\"
#r @"MathNet.Numerics.Signed.3.17.0\lib\net40\MathNet.Numerics.dll"
#r @"MathNet.Numerics.FSharp.Signed.3.17.0\lib\net40\MathNet.Numerics.FSharp.dll"
#load @"FSharp.Charting.0.90.14\FSharp.Charting.fsx"

open System
open MathNet.Numerics.Distributions
open MathNet.Numerics.Statistics
open FSharp.Charting

let calcMean (data:seq<float>) = Statistics.Mean(data)
let line (min:float)
         (max:float)
         (data:seq<float*float>) =
    Chart.Line(data).WithXAxis(Min=min,Max=max)
let histogram (min:float)
              (max:float)
              (data:seq<float>) =
    Chart.Histogram(data, LowerBound = min, UpperBound=max)
let sqrt x = System.Math.Sqrt x

let min = 0.
let max = 1.
let sampleSize = 1000
let stepSize = 0.001
// let lambda = 1.
let variance = 1./8. // 1./(lambda**2.)
let mean = 0.5 // 1./lambda
let initialDistribution = Beta(0.5, 0.5) // Exponential(lambda)
let hist (data:seq<float>) = histogram min max data
let lin (data:seq<float*float>) = line min max data
    
let samples = initialDistribution.Samples()
              |> Seq.take sampleSize
let pdf = seq{ (min + stepSize) .. stepSize .. (max - stepSize) }
          |> Seq.map(fun n -> n,initialDistribution.Density(n))

Chart.Rows([hist samples; lin pdf])

let buildRange (distribution:IContinuousDistribution)
               (size:int)
               (rangeSize:int) =
    seq { 0 .. rangeSize }
    |> Seq.map (fun i ->
        distribution.Samples()
        |> Seq.take size
        |> calcMean)

let buildNormalDistribution (mean:float)
                            (variance:float)
                            (min:float)
                            (max:float)
                            (size:int)
                            (step:float) =
    let n = Normal(mean, Math.Sqrt(variance/(float size)))
    seq { min .. step .. max }
    |> Seq.map (fun x -> x,n.Density(x))

let assembleData (size:int) =
    buildRange initialDistribution size sampleSize,
    buildNormalDistribution mean variance min max size stepSize

let fiveRange,fiveNormal = assembleData 5
Chart.Rows([ hist fiveRange; lin fiveNormal ])

let tenRange,tenNormal = assembleData 10
Chart.Rows([ hist tenRange; lin tenNormal ])

let fiftyRange,fiftyNormal = assembleData 50
Chart.Rows([ hist fiftyRange; lin fiftyNormal ])
