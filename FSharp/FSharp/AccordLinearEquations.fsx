#I @"..\packages\"
#r @"Accord.3.4.0\lib\net46\Accord.dll"
#r @"Accord.Math.3.4.0\lib\net46\Accord.Math.dll"

open System.IO
open Accord
open Accord.Math

let resultPath = __SOURCE_DIRECTORY__ + @"..\..\..\Results\submission_fs-2.txt"

let func (x:float) = sin(x/5.) * exp(x/10.) + 5. * exp(-x/2.)

let buildEquation func ((size:int),(points:float[])) =
    let equationMatrix, equationVector =
        points
        |> Array.map(fun point ->
            ([|0..size|] |> Array.map (fun pow -> point**(float pow)),
             func(point)))
        |> Array.unzip
    (equationMatrix |> array2D, equationVector)
let solveEquation ((a:float[,]),(b:float[])) = a.Solve b
let buildResult = Array.map (fun (r:float) -> r.ToString()) >> String.concat " "

let results =
    [|(1, [|1.;15.|]); (2,[|1.;8.;15.|]); (3,[|1.;4.;10.;15.|])|]
    |> Array.map (buildEquation func >> solveEquation >> buildResult)

results |> Array.iter (printfn "%s")

let resultWriter = File.CreateText resultPath
results |> Array.last |> resultWriter.Write
resultWriter.Close()
