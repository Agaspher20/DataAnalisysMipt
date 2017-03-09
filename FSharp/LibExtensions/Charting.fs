namespace LibExtensions

open FSharp.Charting
module Charting = 
    let maxX max (chart:ChartTypes.GenericChart) =
        chart.WithXAxis(Max=max)
    let maxY max (chart:ChartTypes.GenericChart) =
        chart.WithYAxis(Max=max)
    let minX min (chart:ChartTypes.GenericChart) =
        chart.WithXAxis(Min=min)
    let minY min (chart:ChartTypes.GenericChart) =
        chart.WithYAxis(Min=min)
    let titleX title (chart:ChartTypes.GenericChart) =
        chart.WithXAxis(Title=title)
    let titleY title (chart:ChartTypes.GenericChart) =
        chart.WithYAxis(Title=title)
    let withoutXAxis (chart:ChartTypes.GenericChart) =
        chart.WithXAxis(Enabled=false)
    let withoutYAxis (chart:ChartTypes.GenericChart) =
        chart.WithYAxis(Enabled=false)
    let namedAs name (chart:ChartTypes.GenericChart) =
        chart.WithStyling(Name=name)
    let withTitle title (chart:ChartTypes.GenericChart) = chart.WithTitle title
    let xLabelsFontSize fontSize (chart:ChartTypes.GenericChart) =
        chart.WithXAxis(LabelStyle=ChartTypes.LabelStyle(FontSize=fontSize))
    let yLabelsFontSize fontSize (chart:ChartTypes.GenericChart) =
        chart.WithYAxis(LabelStyle=ChartTypes.LabelStyle(FontSize=fontSize))

    let pairPlot (titles:string[]) (data:seq<float>[]) =
        if titles.Length <> data.Length
        then invalidArg "Titles and data length must be equal" "data"

        let pairPlotData =
            titles
            |> Array.Parallel.mapi (fun i t ->
                let column = data.[i]
                (t,column,column|>Seq.max,column|>Seq.min))
        let lastRowIndex = titles.Length - 1

        pairPlotData
        |> Array.Parallel.mapi (fun i (ti,di,maxi,mini) ->
            pairPlotData
            |> Array.Parallel.mapi (fun j (tj,dj,maxj,minj) ->
                let setBounds = maxX maxj >> maxY maxi >> minX minj >> minY mini
                let ch = match i=j with
                         | true -> dj |> Chart.Histogram
                         | false -> (dj,di)
                                    ||> Seq.zip
                                    |> Chart.Point
                                    |> setBounds
                let titledCh = match j with
                               | 0 -> match i with
                                      | idx when idx = lastRowIndex -> ch |> titleY ti |> titleX tj
                                      | _ -> ch |> titleY ti |> withoutXAxis
                               | _ -> match i with
                                      | idx when idx = lastRowIndex -> ch |> titleX tj |> withoutYAxis
                                      | _ -> ch |> withoutXAxis |> withoutYAxis
                titledCh |> xLabelsFontSize 0.01 |> yLabelsFontSize 0.01)
            |> Chart.Columns)
        |> Chart.Rows