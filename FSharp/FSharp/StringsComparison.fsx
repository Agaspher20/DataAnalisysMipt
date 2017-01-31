#I @"..\packages\"
#r @"MathNet.Numerics.Signed.3.17.0\lib\net40\MathNet.Numerics.dll"
#r @"MathNet.Numerics.FSharp.Signed.3.17.0\lib\net40\MathNet.Numerics.FSharp.dll"

open System.IO
open System.Text.RegularExpressions
open MathNet.Numerics

let filePath = __SOURCE_DIRECTORY__ + @"..\..\..\Data\sentences.txt"
let resultPath = __SOURCE_DIRECTORY__ + @"..\..\..\Results\submission-1.txt"

let cosine (line0:float[]) (line1:float[]) = Distance.Cosine(line0, line1)

let tokenize (line:string) =
    line.ToLowerInvariant()
    |> Regex(@"[a-z]+").Matches
    |> Seq.cast<Match>
    |> Seq.map (fun m -> m.Value)
    |> Seq.countBy(fun token -> token)

let lines = 
    filePath
    |> File.ReadLines
    |> Seq.map (fun line -> (line, line |> tokenize))
    |> Seq.toArray

let allTokens =
    lines
    |> Seq.collect (fun (_,tokens) -> tokens |> Seq.map (fun (key,_) -> key))
    |> Set.ofSeq

let tokensByLines =
    lines
    |> Seq.map (fun (line,tokens) ->
        let tokensMap = tokens |> Map.ofSeq
        (line,allTokens
        |> Seq.map (fun token ->
            match tokensMap |> Map.tryFind token with
            | None -> 0.
            | Some count -> (float count))
        |> Seq.toArray))
    |> Seq.toList

let firstLine = snd tokensByLines.[0]
let distanceToFirstLine = cosine firstLine

let results =
    tokensByLines.[1..]
    |> List.mapi (fun index (line, vector) -> (index+1, line, vector |> distanceToFirstLine))
    |> List.sortBy (fun (_,_,distance) -> distance)
    |> List.take 2
let result = results |> List.map (fun (index,_,_) -> index.ToString()) |> String.concat " "

printfn "Matrix shape: %i %i" (tokensByLines |> List.length) (firstLine |> Array.length)
printfn "First sentence: \"%s\"\n" (fst lines.[0])

results
|> List.iter (fun (index, line, distance) ->
    printfn "Distance: %f Index: %i\n%s\n" distance index line)

printfn "Result: %s" result

let resultWriter = resultPath |> File.CreateText
result |> resultWriter.Write
resultWriter.Close()
