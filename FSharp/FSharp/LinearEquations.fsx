#I @"..\packages\"
#r @"MathNet.Numerics.Signed.3.17.0\lib\net40\MathNet.Numerics.dll"
#r @"MathNet.Numerics.FSharp.Signed.3.17.0\lib\net40\MathNet.Numerics.FSharp.dll"

open System.IO
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra

let resultPath = __SOURCE_DIRECTORY__ + @"..\..\..\Results\submission-2.txt"

let func (x:float) = sin(x/5.) * exp(x/10.) + 5. * exp(-x/2.)
    
let buildEquation func ((size:int),(points:List<float>)) =
    let equationMatrix, equationVector =
        points
        |> List.map(fun point ->
            ([0..size] |> List.map (fun pow -> point**(float pow)),
             func(point)))
        |> List.unzip
    (matrix equationMatrix, vector equationVector)

let buildResult = Seq.map (fun r -> r.ToString()) >> String.concat " "
let solveEquation ((a:Matrix<float>),(b:Vector<float>)) = a.Solve b

let results =
    [|(1, [1.;15.]); (2,[1.;8.;15.]); (3,[1.;4.;10.;15.])|]
    |> Array.map (buildEquation func >> solveEquation >> buildResult)

results |> Array.iter (printfn "%s")

let resultWriter = File.CreateText resultPath
results |> Array.tail |> Array.iter resultWriter.Write
resultWriter.Close()
