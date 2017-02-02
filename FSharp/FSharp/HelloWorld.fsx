let x = 5
let y:int = 4

printf "%s\n" <| x.GetType().Name

let typeName value = value.GetType().Name

printfn "%s" <| typeName y

let a = 4 + 5
let b = 4 * 5
let c = 5/4

printfn "%i" <| -5/4
printfn "%i" <| -(5/4)

let x = 5 * 1000000 * 1000000 * 1000000 * 1000000 + 1
printfn "%i" x
printfn "%s" <| typeName x

let x = 5 * 1000000 * 1000000 * 1000000 * 1000000 * 1000000 * 1000000 * 1000000 * 1000000 * 1000000 * 1000000 + 1
printfn "%i" x
printfn "%s" <| typeName x

let x = 5L * 1000000L * 1000000L * 1000000L + 1L

let mutable y = 5L
y <- x
printfn "%i" y

let y = 5.7
printfn "%f" y
printfn "%s" <| typeName y
y |> typeName |> printfn "%s"

let a = 4.2 + 5.1
let b = 4.2 * 5.1
let c = 5.0 / 4.0
let d = 7. * 8.

let a = 5
let b = 4
float a / float b |> printfn "%f"

let a:bool = true
let b = false

a |> printfn "%b"
a |> typeName |> printfn "%s"

b |> printfn "%b"
b |> typeName |> printfn "%s"

let x = a && b
let x = a && a
let x = b && b

printfn "%i" <| a.GetHashCode()

let z = a.GetHashCode()

printfn "%i" z

printfn "%b" (true && false)
printfn "%b" (true && true)
printfn "%b" (false && true)

printfn "%b" (true || false)
printfn "%b" (true || true)
printfn "%b" (false || false)

let z = ()
printfn "%s" (z.ToString())
printfn "%s" <| typeName z

let x = "abc"
printfn "%s" x
x |> typeName |> printfn "%s"

let a = "Ivan"
let b = "Ivanov"
let s = a + " " + b

a.ToUpper() |> printfn "%s"
b.ToLower() |> printfn "%s"

a.Length |> printfn "%i"

a.[0] |> printfn "%c"
let part = a.[0..2]
part |> typeName |> printfn "%s"
a |> String.iter (printfn "%c")
a |> String.iter (fun value -> printfn "%c" value)
a |> String.iter (fun value -> value |> printfn "%c")
a |> String.iter (fun value -> value |> typeName |> printfn "%s")
a.[0..2] |> String.iter (printfn "%c")

let goods = [33450; 34010; 33990; 33200]
goods |> typeName |> printfn "%s"
goods.Length |> printfn "%i"
let arrayGoods = [|33450; 34010; 33990; 33200|]
arrayGoods |> typeName |> printfn "%s"
let tupleGoods = 33450, 34010, 33990, 33200
tupleGoods |> typeName |> printfn "%s"

goods |> List.map (fun good -> good.ToString()) |> String.concat "---" |> printfn "%s"

let features = "Ivan Ivanovich", "Medium", 50000, 12, true
let (name,profession,earn,_,_) = features
printfn "%s %s %i" name profession earn

type Features = { Name: string; Profession: string; Earn: int; Age:int; Something:bool}
features |> typeName |> printfn "%s"

let otherFeatures = { Name = "Sergey Sergeevich"; Profession = "Administrator"; Earn = 5*100000; Age = 42; Something = false}
otherFeatures |> typeName |> printfn "%s"

arrayGoods |> Array.iter (printf "%i ")
arrayGoods
|> Array.map (fun value -> value.ToString())
|> String.concat " "
|> printfn "%s"

let goods = List.append goods [8]
let arrayGoods = Array.append arrayGoods [|7|]
let goodsSet = Set.ofArray arrayGoods
let goodsSet = Set.add 6 goodsSet
goodsSet |> Set.add 9
goodsSet


let testA   = float 2
let testB x = float 2
let testC x = float 2 + x
let testD x = x.ToString().Length
let testE (x:float) = x.ToString().Length
let testF x = printfn "%s" x
let testG x = printfn "%f" x
let testH   = 2 * 2 |> ignore
let testI x = 2 * 2 |> ignore
let testJ (x:int) = 2 * 2 |> ignore
let testK   = "hello"
let testL() = "hello"
let testM x = x=x
let testN x = x 1          // hint: what kind of thing is x?
let testO x:string = x 1   // hint: what does :string modify? 

let add1 = (+) 1
let toString x = x.ToString()
let myFunc x y =
    printfn "X is %s" x
    printfn "Y is %s" y

3 |> add1 |> toString |> myFunc <| (4 |> add1 |> toString)

let add n x = x + n
let add1ThenMultiply = (+) 1 >> (*) 
let times n x = x * n
let times2Add1 = add 1 << times 2
