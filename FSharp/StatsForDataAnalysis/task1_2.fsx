#r @"..\packages\FSharp.Data.2.3.2\lib\net40\FSharp.Data.dll"
#r @"..\packages\MathNet.Numerics.Signed.3.17.0\lib\net40\MathNet.Numerics.dll"
#r @"..\packages\MathNet.Numerics.FSharp.Signed.3.17.0\lib\net40\MathNet.Numerics.FSharp.dll"

#r @"bin\Debug\LibExtensions.dll"

open System

open FSharp.Data

open MathNet.Numerics
open MathNet.Numerics.Statistics
open MathNet.Numerics.Distributions

open LibExtensions.Stats

// Большая часть млекопитающих неспособны во взрослом возрасте переваривать лактозу, содержащуюся в молоке.
// У людей за расщепление лактозы отвечает фермент лактаза, кодируемый геном LCT.
// У людей с вариантом 13910T этого гена лактаза продолжает функционировать на протяжении всей жизни.
// Распределение этого варианта гена сильно варьируется в различных генетических популяциях.
// Из 50 исследованных представителей народа майя вариант 13910T был обнаружен у одного.
// Постройте нормальный 95% доверительный интервал для доли носителей варианта 13910T в популяции майя.
// Чему равна его нижняя граница? Округлите ответ до 4 знаков после десятичной точки.
let n = 50
let nSuccess = 1
let alpha = 0.05

let proportion_confidence_interval alpha (successCount:int) (count:int) =
    let successLength = successCount |> float
    let dataLength = count |> float
    let p = successLength/dataLength
    let z = alpha/2.*dataLength
    let standardError = sqrt(p*(1.-p)/dataLength)
    (p-z*standardError,p+z*standardError)

(nSuccess, n)
||> (proportion_confidence_interval alpha)
||> (printfn "Confidence interval: [%.4f;%.4f]")
