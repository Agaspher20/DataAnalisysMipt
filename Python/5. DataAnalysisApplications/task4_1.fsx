let r1 = [|1.;0.;0.;1.|]
let r2 = [|1.;1.;0.;0.|]

let precision (res: float seq) =
    (res |> Seq.sum)/(res |> Seq.length |> float)

let averagePrecision (res: float seq) =
    let s1 = res
             |> Seq.mapi (fun i r -> r * (res |> Seq.take (i + 1) |> precision))
             |> Seq.sum
    let s2 = res |> Seq.sum
    s1/s2

let result1 = r1 |> averagePrecision
let result2 = r2 |> averagePrecision

System.Math.Abs(result1-result2) |> (printfn "%.2f")
