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
