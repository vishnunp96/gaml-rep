using ProgressMeter
using Distributed

p = Progress(100)
channel = RemoteChannel(()->Channel{Bool}(10), 1)

@sync begin
    # this task prints the progress bar
    @async while take!(channel)
        next!(p)
    end

    # this task does the computation
    @async begin
        @distributed (+) for i in 1:100
            sleep(0.5)
            put!(channel, true)
            i^2
        end
        put!(channel, false) # this tells the printing task to finish
    end
end
