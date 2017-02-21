#I @"..\packages\"
#r @"Accord.3.4.0\lib\net46\Accord.dll"
#r @"Accord.Math.3.4.0\lib\net46\Accord.Math.dll"
#load @"FSharp.Charting.0.90.14\FSharp.Charting.fsx"

open System
open System.IO
open Accord
open Accord.Math
open Accord.Math.Optimization
open FSharp.Charting

let resultPath = __SOURCE_DIRECTORY__ + @"..\..\..\Results\submission_fs-1.txt"

let func (x:float) = sin(x/5.) * exp(x/10.) + 5. * exp(-x/2.)
let gradientFunc (x:float) = cos(x/5.) * exp(x/10.)/5.+ sin(x/5.) * exp(x/10.)/10. - 5.*exp(-x/2.)/2.

let bfgs (fn:float->float) (grad:float->float) =
    let bfgsFn (x:float[]) = fn x.[0]
    let bfgsGr (x:float[]) = [|grad x.[0]|]
    let solver = 
        BroydenFletcherGoldfarbShanno(
            1,
            new Func<float[], float>(bfgsFn),
            new Func<float[], float[]>(bfgsGr))
    solver

let minimizer = bfgs func gradientFunc

let result =
    [|2.;30.|]
    |> Array.map (fun point ->
        let r = minimizer.Minimize([|point|])
        minimizer.Value,minimizer.Solution)
    |> Array.map (fun (v,_) -> Math.Round(v,2).ToString())
    |> String.concat " "

printfn "%s" result

let resultWriter = File.CreateText resultPath
resultWriter.Write result
resultWriter.Close()

Chart.Line ({0..30} |> Seq.map (fun point -> point |> float |> func))
