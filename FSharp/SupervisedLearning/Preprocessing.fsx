#r @"..\packages\Accord.3.4.0\lib\net46\Accord.dll"
#r @"..\packages\Accord.Math.3.4.0\lib\net46\Accord.Math.dll"
#r @"..\packages\Accord.Math.3.4.0\lib\net46\Accord.Math.Core.dll"
#r @"..\packages\Accord.MachineLearning.3.4.0\lib\net46\Accord.MachineLearning.dll"
#r @"..\packages\Accord.Statistics.3.4.0\lib\net46\Accord.Statistics.dll"
#r @"..\packages\FSharp.Data.2.3.2\lib\net40\FSharp.Data.dll"
#r @"..\packages\FSharp.Charting.0.90.14\lib\net40\FSharp.Charting.dll"
#r @"bin\Debug\LibExtensions.dll"

open System.Data

open Accord
open Accord.Math
open Accord.MachineLearning
open Accord.Statistics.Filters
open Accord.Statistics.Models.Regression
open Accord.Statistics.Models.Regression.Fitting

open FSharp.Data
open FSharp.Charting

open LibExtensions.Charting

[<Literal>]
let DataPath = __SOURCE_DIRECTORY__ + @"..\..\..\Data\preprocessing_data.csv"
type Data = CsvProvider<DataPath>
type Observation = Data.Row
let dataSet = Data.Load(DataPath)
let getNumericValues (o:Observation) =
    [|o.``RFCD.Percentage.1``;
     o.``RFCD.Percentage.2``;
     o.``RFCD.Percentage.3``;
     o.``RFCD.Percentage.4``;
     o.``RFCD.Percentage.5``;
     o.``SEO.Percentage.1``;
     o.``SEO.Percentage.2``;
     o.``SEO.Percentage.3``;
     o.``SEO.Percentage.4``;
     o.``SEO.Percentage.5``;
     o.``Year.of.Birth.1``;
     o.``Number.of.Successful.Grant.1``;
     o.``Number.of.Unsuccessful.Grant.1``|]
let getCategoricalValues (o:Observation) =
    [|string o.``Country.of.Birth.1``;
     string o.``SEO.Code.1``;
     string o.``SEO.Code.2``;
     string o.``SEO.Code.3``;
     string o.``SEO.Code.4``;
     string o.``SEO.Code.5``;
     string o.``RFCD.Code.1``;
     string o.``RFCD.Code.2``;
     string o.``RFCD.Code.3``;
     string o.``RFCD.Code.4``;
     string o.``RFCD.Code.5``;
     string o.``A..1``;
     string o.``A.1``;
     string o.``B.1``;
     string o.``C.1``;
     string o.``No..of.Years.in.Uni.at.Time.of.Grant.1``;
     string o.``With.PHD.1``;
     string o.``Sponsor.Code``;
     string o.``Home.Language.1``;
     string o.``Person.ID.1``;
     string o.``Grant.Category.Code``;
     string o.``Faculty.No..1``;
     string o.``Role.1``;
     string o.``Dept.No..1``;
     string o.``Contract.Value.Band...see.note.A``|]
let transformCategoricalValues (rows:string[] seq) =
    let table = new DataTable("TempTable")
    let buildColumn name = new DataColumn(name, typedefof<string>)
    table.Columns.AddRange(
        [|
            buildColumn "Country.of.Birth.1";
            buildColumn "SEO.Code.1";
            buildColumn "SEO.Code.2";
            buildColumn "SEO.Code.3";
            buildColumn "SEO.Code.4";
            buildColumn "SEO.Code.5";
            buildColumn "RFCD.Code.1";
            buildColumn "RFCD.Code.2";
            buildColumn "RFCD.Code.3";
            buildColumn "RFCD.Code.4";
            buildColumn "RFCD.Code.5";
            buildColumn "A..1";
            buildColumn "A.1";
            buildColumn "B.1";
            buildColumn "C.1";
            buildColumn "No..of.Years.in.Uni.at.Time.of.Grant.1";
            buildColumn "With.PHD.1";
            buildColumn "Sponsor.Code";
            buildColumn "Home.Language.1";
            buildColumn "Person.ID.1";
            buildColumn "Grant.Category.Code";
            buildColumn "Faculty.No..1";
            buildColumn "Role.1";
            buildColumn "Dept.No..1";
            buildColumn "Contract.Value.Band...see.note.A"
        |])
    rows |> Seq.iter(fun row ->
        let tableRow = table.NewRow()
        row |> Array.iteri (fun i c -> tableRow.[i] <- c)
        table.Rows.Add(tableRow) |> ignore)
    let codebook = Codification(table)
    let resultTable = codebook.Apply(table)
    Matrix.ToJagged(resultTable)
let calculateMeans (rows:float[] seq) =
    let columnsCount = (rows |> Seq.head |> Array.length) - 1
    [0 .. columnsCount]
    |> List.map(fun colIdx ->
        let nonZero =
            rows
            |> Seq.map(fun row -> row.[colIdx])
            |> Seq.filter (fun col -> match col with
                                      | x when nan.Equals(x) -> false
                                      | x when x = 0. -> false
                                      | _ -> true)
            |> Seq.toList
        let correction = nonZero |> List.max
        let colCount = nonZero |> List.length |> float
        let sum = nonZero |> List.sumBy(fun col -> col/correction)
        sum*correction/colCount)
dataSet.Rows
|> Seq.take 5
|> Seq.iter (fun o ->
    (o|>getNumericValues|>Array.map string,o|>getCategoricalValues)
    ||> Array.append
    |> (printfn "%A"))
let numericData = dataSet.Rows
                  |> Seq.map getNumericValues
let numericMeans = numericData |> calculateMeans
let mapRowValues (data:float[] seq) mapper =
    data
    |> Seq.map (fun values ->
        values
        |> Array.mapi(fun i x -> match x with
                                 | v when nan.Equals(v) -> v |> mapper i
                                 | v -> v))
    |> Seq.toArray
let mapNumericData = mapRowValues numericData
let xRealZeros = mapNumericData (fun i v -> 0.)
let xRealMean = mapNumericData (fun i v -> numericMeans.[i])
let xCategoricalData = dataSet.Rows
                      |> Seq.map (getCategoricalValues
                                  >> (fun row ->
                                      row
                                      |> Array.map (fun str ->
                                        match str with
                                        | null -> "<null>"
                                        | "" -> "<empty>"
                                        | _ -> str)))
                      |> transformCategoricalValues
let y = dataSet.Rows
        |> Seq.map (fun row -> row.``Grant.Status``)
        |> Seq.toArray

let zeroCat = 
    (xRealZeros, xCategoricalData)
    ||> Array.map2 (fun r c -> (r,c) ||> Array.append)

let meanCat =
    (xRealMean, xCategoricalData)
    ||> Array.map2 (fun r c -> (r,c) ||> Array.append)

let crossVal = CrossValidation<float[]>(y |> Array.length)
crossVal.Fitting <- CrossValidationFittingFunction<float[]>((fun k indicesTrain indicesValidation ->
    let trainY = y.Get(indicesTrain)
    let validationY = y.Get(indicesValidation)

    let trainZeroCat = zeroCat.Get(indicesTrain)
    let validationZeroCat = zeroCat.Get(indicesValidation)

    let learner = new IterativeReweightedLeastSquares<LogisticRegression>()
    learner.Tolerance <- 1e-4
    learner.Iterations <- 100
    learner.Regularization <- 0.

    let regression = learner.Learn(trainZeroCat, trainY)
    let calcError (inputs:float[][]) outputs =
        let errSum =
            (inputs, outputs)
            ||> Array.map2 (fun input output ->
                                match regression.Decide(input)=output with
                                | true -> 1.
                                | false -> 0.)
            |> Array.sum
        errSum/(outputs |> Array.length |> float)
    // todo: GridSearch
    // todo: LogisticRegressionAnalysis 
    CrossValidationValues<float[]>(
        calcError trainZeroCat trainY,
        calcError validationZeroCat validationY)))

let res = crossVal.Compute()

printfn "%A" (res.Training.Values.ArgMin(),res.Validation.Values.ArgMin())