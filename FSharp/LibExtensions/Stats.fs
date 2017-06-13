namespace LibExtensions

open MathNet.Numerics
open MathNet.Numerics.Statistics
open MathNet.Numerics.Distributions

module Stats = 
    let confidenceInterval (alpha:float) (data:seq<float>) =
        let mean = Statistics.Mean(data)
        let std = Statistics.StandardDeviation(data)
        let length = (data |> Seq.length |> float)
        let t = StudentT.InvCDF(0., 1., length - 1., alpha/2.)
        let T = -t * (std / sqrt(length))
        (mean-T,mean+T)
