using Random, Statistics, LinearAlgebra

function gen_ts(n)
    cumsum(randn(n))
end

znorm(x) = (x .- mean(x)) ./ std(x)

function avg_dist(ts, w; samples=1000000)
    n = length(ts)
    s = 0
    for (i, j) in zip(rand(1:n-w, samples), rand(1:n-w, samples))
        a = @view ts[i:(i+w-1)]
        b = @view ts[j:(j+w-1)]
        s += norm(znorm(a) .- znorm(b))
    end
    s / samples
end

function inject(ts, w, rc)
    n = length(ts)
    avg = avg_dist(ts, w)
    base = sin.(1:w)
    scale = 0.0001
    a = base .+ scale .* randn(w)
    b = base .+ scale .* randn(w)
    ab_rc =  avg / norm(znorm(a) .- znorm(b))
    while ab_rc > rc
        scale *= 2
        a = base .+ scale .* randn(w)
        b = base .+ scale .* randn(w)
        ab_rc =  avg / norm(znorm(a) .- znorm(b))
    end
    i = rand(1:n-w)
    j = rand(1:n-w)
    ts[i:i+w-1] = a .+ ts[i]
    ts[j:j+w-1] = b .+ ts[j]
    @info "motif implanted" i j norm(znorm(a) .- znorm(b)) ab_rc
    ts
end

for n in 10 .^ [5,6,7,8,9]
    w = 100
    for rc in [1000, 100]
        ts = inject(gen_ts(n), w, rc)
        if rc == 1000
            difficulty = "easy"
        else
            difficulty = "difficult"
        end
        open("data/synth-$(difficulty)-$(n).txt", "w") do fp
            for x in ts
                println(fp, x)
            end
        end
    end
end
