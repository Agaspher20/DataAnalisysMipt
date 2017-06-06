#r @"..\packages\FSharp.Data.2.3.2\lib\net40\FSharp.Data.dll"
#r @"..\packages\FSharp.Charting.0.90.14\lib\net40\FSharp.Charting.dll"
#r @"..\packages\MathNet.Numerics.Signed.3.17.0\lib\net40\MathNet.Numerics.dll"
#r @"..\packages\MathNet.Numerics.FSharp.Signed.3.17.0\lib\net40\MathNet.Numerics.FSharp.dll"

open System

open FSharp.Data
open FSharp.Charting

open MathNet.Numerics
open MathNet.Numerics.Statistics
open MathNet.Numerics.Distributions

[<Literal>]
let DataPath = __SOURCE_DIRECTORY__ + @"..\..\..\Data\water.txt"

type Data = CsvProvider<DataPath, Schema="string,string,int,int">
type Observation = Data.Row

let dataSet = Data.Load(DataPath)

let formatObservation (obs:Observation) =
    sprintf "%-8s %-10s %9i %8i"
            obs.Location
            obs.Town
            obs.Mortality
            obs.Hardness

let headers = match dataSet.Headers with
                | Some h -> sprintf "%-8s %-10s %-9s %-8s" h.[0] h.[1] h.[2] h.[3]
                | None -> String.Empty

printfn "%s" headers
dataSet.Rows
|> Seq.take 5
|> Seq.iter (formatObservation >> (printfn "%s"))

// Для 61 большого города в Англии и Уэльсе известны средняя годовая смертность на 100000 населения
// (по данным 1958–1964) и концентрация кальция в питьевой воде (в частях на миллион). Чем выше концентрация
// кальция, тем жёстче вода. Города дополнительно поделены на северные и южные.
//
// Постройте 95% доверительный интервал для средней годовой смертности в больших городах. Чему равна его нижняя
// граница? Округлите ответ до 4 знаков после десятичной точки. 

let confidenceInterval (alpha:float) (data:seq<float>) =
    let mean = Statistics.Mean(data)
    let std = Statistics.StandardDeviation(data)
    let length = (data |> Seq.length |> float)
    let t = StudentT.InvCDF(0., 1., length - 1., alpha/2.)
    let T = -t * (std / sqrt(length))
    (mean-T,mean+T)

dataSet.Rows
|> Seq.map ((fun obs -> obs.Mortality) >> float)
|> (confidenceInterval 0.05)
||> (printfn "Mortality confidence interval: [%.4f;%.4f]")
dataSet.Rows
|> Seq.map ((fun obs -> obs.Mortality) >> float)
|> (fun mortalities -> Statistics.Mean(mortalities))
|> (printfn "Mortality mean: %.4f")

// На данных из предыдущего вопроса постройте 95% доверительный интервал для средней годовой смертности по всем
// южным городам. Чему равна его верхняя граница? Округлите ответ до 4 знаков после десятичной точки.
dataSet.Rows
|> Seq.filter (fun obs -> obs.Location = "South")
|> Seq.map ((fun obs -> obs.Mortality) >> float)
|> (confidenceInterval 0.05)
||> (printfn "Mortality for south cities confidence interval: [%.4f;%.4f]")
dataSet.Rows
|> Seq.filter (fun obs -> obs.Location = "South")
|> Seq.map ((fun obs -> obs.Mortality) >> float)
|> (fun mortalities -> Statistics.Mean(mortalities))
|> (printfn "Mortality for south cities mean: %.4f")

// На тех же данных постройте 95% доверительный интервал для средней годовой смертности по всем северным городам.
// Пересекается ли этот интервал с предыдущим? Как вы думаете, какой из этого можно сделать вывод? 
dataSet.Rows
|> Seq.filter (fun obs -> obs.Location = "North")
|> Seq.map ((fun obs -> obs.Mortality) >> float)
|> (confidenceInterval 0.05)
||> (printfn "Mortality for north cities confidence interval: [%.4f;%.4f]")
dataSet.Rows
|> Seq.filter (fun obs -> obs.Location = "North")
|> Seq.map ((fun obs -> obs.Mortality) >> float)
|> (fun mortalities -> Statistics.Mean(mortalities))
|> (printfn "Mortality for north cities mean: %.4f")
// Интервалы не пересекаются; видимо, средняя смертность на севере и на юге существенно разная 

// Пересекаются ли 95% доверительные интервалы для средней жёсткости воды в северных и южных городах?
dataSet.Rows
|> Seq.filter (fun obs -> obs.Location = "South")
|> Seq.map ((fun obs -> obs.Hardness) >> float)
|> (confidenceInterval 0.05)
||> (printfn "Water hardness confidence interval for south cities: [%.4f;%.4f]")
dataSet.Rows
|> Seq.filter (fun obs -> obs.Location = "North")
|> Seq.map ((fun obs -> obs.Hardness) >> float)
|> (confidenceInterval 0.05)
||> (printfn "Water hardness confidence interval for north cities: [%.4f;%.4f]")
// Не пересекаются

// Вспомним формулу доверительного интервала для среднего нормально распределённой случайной величины
// с дисперсией sigma^2
// При σ=1 какой нужен объём выборки, чтобы на уровне доверия 95% оценить среднее с точностью ±0.1?
let z = 1.95996
printfn "%f" ((z/0.1)**2.)
