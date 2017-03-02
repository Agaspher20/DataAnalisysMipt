#r @"..\packages\Accord.3.4.0\lib\net46\Accord.dll"
#r @"..\packages\Accord.Statistics.3.4.0\lib\net46\Accord.Statistics.dll"
#r @"..\packages\FSharp.Charting.0.90.14\lib\net40\FSharp.Charting.dll"
#r @"bin\Debug\LibExtensions.dll"

open Accord
open Accord.Statistics
open Accord.Statistics.Analysis

open FSharp.Charting
open LibExtensions.Charting

let scatter (actual:float seq) (predicted:float seq) t =
    [
        (actual,predicted)
        ||> Seq.zip
        |> Chart.Point;
        [(-1.,t);(2.,t)]
        |> Chart.Line
    ]
    |> Chart.Combine
    |> minX -1.
    |> maxX 2.
    |> titleY "Predicted"
    |> titleX "Actual"

let manyScatters actuals predicteds ts labels columnsCount =
    (actuals,predicteds,ts)
    |||> Seq.zip3
    |> Seq.mapi(fun i (actual,predicted,t) -> i,scatter actual predicted t)
    |> Seq.zip labels
    |> Seq.map(fun (lbl,(i,chart)) -> i,chart |> withTitle lbl)
    |> Seq.groupBy(fun (i,_) -> i/columnsCount)
    |> Seq.map(fun (_,plots) -> plots |> Seq.map snd |> Chart.Columns)
    |> Chart.Rows

let perfectActual = [ 0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  
                     1.;  1.;  1.; 1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.]
let perfectPredicted = [ 0.19015288;  0.23872404;  0.42707312;  0.15308362;  0.2951875 ;
                        0.23475641;  0.17882447;  0.36320878;  0.33505476;  0.202608  ;
                        0.82044786;  0.69750253;  0.60272784;  0.9032949 ;  0.86949819;
                        0.97368264;  0.97289232;  0.75356512;  0.65189193;  0.95237033;
                        0.91529693;  0.8458463 ]
scatter perfectActual perfectPredicted 0.5
|> withTitle "Perfect algorithm"
|> namedAs "Perfect algorithm"
|> Chart.Show

let typicalActual = [ 0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;
                       0.;  0.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;
                       1.;  1.;  1.;  1. ]
let typicalPredicted = [ 0.41310733;  0.43739138;  0.22346525;  0.46746017;  0.58251177;
                          0.38989541;  0.43634826;  0.32329726;  0.01114812;  0.41623557;
                          0.54875741;  0.48526472;  0.21747683;  0.05069586;  0.16438548;
                          0.68721238;  0.72062154;  0.90268312;  0.46486043;  0.99656541;
                          0.59919345;  0.53818659;  0.8037637 ;  0.272277  ;  0.87428626;
                          0.79721372;  0.62506539;  0.63010277;  0.35276217;  0.56775664 ]
let awfulActual = [ 1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  0.;
                     0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0. ]
let awfulPredicted = [ 0.07058193;  0.57877375;  0.42453249;  0.56562439;  0.13372737;
                        0.18696826;  0.09037209;  0.12609756;  0.14047683;  0.06210359;
                        0.36812596;  0.22277266;  0.79974381;  0.94843878;  0.4742684 ;
                        0.80825366;  0.83569563;  0.45621915;  0.79364286;  0.82181152;
                        0.44531285;  0.65245348;  0.69884206;  0.69455127 ]

manyScatters
    [ perfectActual;typicalActual;awfulActual ]
    [ perfectPredicted;typicalPredicted;awfulPredicted ]
    [ 0.5;0.5;0.5 ]
    [ "Perfect";"Typical";"Awful" ]
    3
|> Chart.Show

let perfectRiskyActual = [ 0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  1.;  1.;
                            1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1. ]
let perfectRiskyPredicted = [ 0.23563765;  0.16685597;  0.13718058;  0.35905335;  0.18498365;
                               0.20730027;  0.14833803;  0.18841647;  0.01205882;  0.0101424 ;
                               0.10170538;  0.94552901;  0.72007506;  0.75186747;  0.85893269;
                               0.90517219;  0.97667347;  0.86346504;  0.72267683;  0.9130444 ;
                               0.8319242 ;  0.9578879 ;  0.89448939;  0.76379055 ]

let typicalRiskyActual = [ 0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  1.;
                            1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1. ]
let typicalRiskyPredicted = [ 0.13832748;  0.0814398 ;  0.16136633;  0.11766141;  0.31784942;
                               0.14886991;  0.22664977;  0.07735617;  0.07071879;  0.92146468;
                               0.87579938;  0.97561838;  0.75638872;  0.89900957;  0.93760969;
                               0.92708013;  0.82003675;  0.85833438;  0.67371118;  0.82115125;
                               0.87560984;  0.77832734;  0.7593189;  0.81615662;  0.11906964;
                               0.18857729 ]

manyScatters [ perfectActual; typicalActual; perfectRiskyActual; typicalRiskyActual ]
             [ perfectPredicted; typicalPredicted; perfectRiskyPredicted; typicalRiskyPredicted ]
             [ 0.5;0.5;0.5;0.5 ]
             [ "Perfect careful"; "Typical careful"; "Perfect risky"; "Typical risky" ]
             2
|> Chart.Show

let avoidsFPActual = [ 0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;
                        0.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;
                        1.;  1.;  1. ]
let avoidsFPPredicted = [ 0.29340574; 0.47340035;  0.1580356 ;  0.29996772;  0.24115457;  0.16177793;
                           0.35552878;  0.18867804;  0.38141962;  0.20367392;  0.26418924; 0.16289102; 
                           0.27774892;  0.32013135;  0.13453541; 0.39478755;  0.96625033;  0.47683139;  
                           0.51221325;  0.48938235; 0.57092593;  0.21856972;  0.62773859;  0.90454639;  0.19406537;
                           0.32063043;  0.4545493 ;  0.57574841;  0.55847795 ]
let avoidsFNActual = [ 0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;  0.;
                        0.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1.;  1. ]
let avoidsFNPredicted = [ 0.35929566; 0.61562123;  0.71974688;  0.24893298;  0.19056711;  0.89308488;
                           0.71155538;  0.00903258;  0.51950535;  0.72153302;  0.45936068;  0.20197229;  0.67092724;
                           0.81111343;  0.65359427;  0.70044585;  0.61983513;  0.84716577;  0.8512387 ;  
                           0.86023125;  0.7659328 ;  0.70362246;  0.70127618;  0.8578749 ;  0.83641841;  
                           0.62959491;  0.90445368 ]

manyScatters
    [ typicalActual;avoidsFPActual;avoidsFNActual ]
    [ typicalPredicted;avoidsFPPredicted;avoidsFNPredicted ]
    [ 0.5;0.5;0.5 ]
    [ "Perfect";"Avoids FP";"Avoids FN" ]
    3
|> Chart.Show

let valuesToBools t (actual,predicted) =
    actual|>Seq.map(fun a -> a > 0.),
    predicted|>Seq.map(fun p -> p > t)
let toConfusionMatrix (actual,predicted) =
    ConfusionMatrix(actual|>Seq.toArray,predicted|>Seq.toArray)

let metricsTable actuals predicteds labels =
    let maxLength = labels |> Seq.map String.length |> Seq.max
    printfn "%*s Precision Recall Accuracy" maxLength System.String.Empty
    (actuals, predicteds)
    ||> Seq.zip
    |> Seq.map (valuesToBools 0.5 >> toConfusionMatrix)
    |> Seq.zip labels
    |> Seq.iter (fun (label,matrix) ->
        printfn "%*s %9.7f %6.4f %8.6f" maxLength label matrix.Precision matrix.Recall matrix.Accuracy)

metricsTable [ perfectActual; typicalActual; awfulActual ]
             [ perfectPredicted; typicalPredicted; awfulPredicted ]
             [ "Perfect"; "Typical"; "Awful" ]

metricsTable [ typicalActual; typicalRiskyActual ]
             [ typicalPredicted; typicalRiskyPredicted ]
             [ "Typical careful"; "Typical risky" ]

metricsTable [ avoidsFPActual; avoidsFNActual ]
             [ avoidsFPPredicted; avoidsFNPredicted ]
             [ "Typical careful"; "Typical risky" ]

let precisionRecallCurve (thresholds:float list) (actual,predicted) =
    thresholds
    |> List.map (fun thr ->
        let mtx = (valuesToBools thr (actual,predicted)) |> toConfusionMatrix
        thr,mtx.Precision,mtx.Recall)
([ typicalActual; avoidsFPActual; avoidsFNActual ],
 [ typicalPredicted; avoidsFPPredicted; avoidsFNPredicted ])
||> List.zip
|> List.map ((precisionRecallCurve [ 0. .. 0.1 .. 1. ])
            >> (fun c -> [
                            c |> List.map(fun (t,p,r) -> (t,p)) |> Chart.Line;
                            c |> List.map(fun (t,p,r) -> (t,r)) |> Chart.Line
                         ])
            >> Chart.Combine
            >> minX 0. >> maxX 1.
            >> minY 0. >> maxY 1.
            >> titleX "threshold")
|> List.zip ["Typical"; "Avoids FP"; "Avoids FN"]
|> List.map (fun (t,ch) -> ch |> withTitle t)
|> Chart.Rows
|> Chart.Show