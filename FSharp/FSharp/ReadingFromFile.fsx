open System.IO

let filePath = __SOURCE_DIRECTORY__ + @"..\..\..\Data\example_reading_files.txt"

let fullFile = File.ReadAllText(filePath)

let allLines = File.ReadAllLines(filePath)

allLines |> Array.iter (printfn "%s")

let allLinesGenerated = [
    use readStream = File.OpenText(filePath)
    while not readStream.EndOfStream do
        yield readStream.ReadLine()
    readStream.Close()
]

allLinesGenerated |> List.iter (printfn "%s")


let lines = seq {
    use fileObject = File.OpenText(filePath)
    let mutable i = 0
    while not fileObject.EndOfStream do
        printfn "%s %i" "Reading line" i
        i <- i + 1
        yield fileObject.ReadLine()
    printfn "%s" "file closed"
    fileObject.Close()
}

lines |> Seq.iter (printfn "%s")

let readLinesLazy path =
    seq {
        use fileObject = File.OpenText(path)
        while not fileObject.EndOfStream do
            yield fileObject.ReadLine()
        fileObject.Close()
    }

let readLinesCondLazy (predicate:int->bool) path =
    seq {
        use fileObject = File.OpenText(path)
        let mutable i = 0
        while not fileObject.EndOfStream do
            match predicate i with
            | true -> yield fileObject.ReadLine()
            | _ -> fileObject.ReadLine() |> ignore
            i <- i+1
                
        fileObject.Close()
    }

filePath |> readLinesLazy |> Seq.iter (printfn "%s")

filePath |> readLinesLazy |> Seq.iteri (fun index line ->
    if index % 2 = 0
    then ()
    else printfn "%s" line)

filePath |> readLinesCondLazy (fun index -> index % 2 = 0) |> Seq.iter (printfn "%s")
