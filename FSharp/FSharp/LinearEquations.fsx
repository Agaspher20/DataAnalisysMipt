#I @"..\packages\"
#r @"MathNet.Numerics.Signed.3.17.0\lib\net40\MathNet.Numerics.dll"
#r @"MathNet.Numerics.FSharp.Signed.3.17.0\lib\net40\MathNet.Numerics.FSharp.dll"

open System
open System.IO
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra

let resultPath = __SOURCE_DIRECTORY__ + @"..\..\..\Results\submission-2.txt"

let func (x:float) =
    Math.Sin(x/5.) * Math.Exp(x/10.) + 5. * Math.Exp(-x/2.)
    
let buildEquation (size:int) (points:List<float>) =
    let equationMatrix =
        points
        |> List.map (fun point ->
            [0..size] |> List.map (fun pow -> Math.Pow(point, float pow)))
    let equationVector = points |> List.map func
    (matrix equationMatrix, vector equationVector)

let buildResult = Seq.map (fun r -> r.ToString()) >> String.concat " "

let a1,b1 = [1.;15.] |> buildEquation 1
a1.Solve b1 |> buildResult |> printfn "%s"

let a2,b2 = [1.;8.;15.] |> buildEquation 2
a2.Solve b2 |> buildResult |> printfn "%s"

let a3,b3 = [1.;4.;10.;15.] |> buildEquation 3
let result = a3.Solve b3 |> buildResult
result |> printfn "%s"

let resultWriter = File.CreateText resultPath
result |> resultWriter.Write
resultWriter.Close()
