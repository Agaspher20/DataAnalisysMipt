open System.IO

let resultPath = __SOURCE_DIRECTORY__ + @"..\..\..\Results\file_to_write_in.txt"

let fileObj = File.CreateText(resultPath)
fileObj.WriteLine("Stroka dlya zapisi v file")
fileObj.Close()

let fileObj = File.CreateText(resultPath)
let secondString = "Drugaya stroka dlya zapisi"
fileObj.WriteLine(secondString)
fileObj.Close()

let fileObj = File.AppendText(resultPath)
let thirdString = "Tret'ya stroka"
fileObj.WriteLine(thirdString)
fileObj.Close()

let fileObj = File.CreateText(resultPath)
[1..10] |> List.iter fileObj.WriteLine
fileObj.Close()
