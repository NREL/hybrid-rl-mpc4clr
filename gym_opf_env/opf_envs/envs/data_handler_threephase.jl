# Data handler

using DataFrames
using CSV

# TYPE DEFINITIONS
mutable struct Generator
   index::Any
   bus_idx::Int
   g_P_max::Float64
   g_Q_max::Float64
   cost::Float64
   function Generator(index, bus_idx, g_P_max, g_S_max, cost)
      g = new()
      g.index  = index
      g.bus_idx = bus_idx
      g.g_P_max = g_P_max
      g.g_Q_max = sqrt(g_S_max^2 - g_P_max^2)
      g.cost = cost
      return g
   end
end

mutable struct Wind
   index::Any
   bus_idx::Int
   w_P_max::Float64
   w_Q_max::Float64
   w_S_max::Float64
   function Wind(index, bus_idx, w_P_max, w_S_max)
      w = new()
      w.index  = index
      w.bus_idx = bus_idx
      w.w_P_max = w_P_max
      w.w_S_max = w_S_max
      w.w_Q_max = sqrt(w_S_max^2 - w_P_max^2)
      return w
   end
end

mutable struct PV
   index::Any
   bus_idx::Int
   p_P_max::Float64
   p_Q_max::Float64
   p_S_max::Float64
   function PV(index, bus_idx, p_P_max, p_S_max)
      p = new()
      p.index  = index
      p.bus_idx = bus_idx
      p.p_P_max = p_P_max
      p.p_S_max = p_S_max
      p.p_Q_max = sqrt(p_S_max^2 - p_P_max^2)
      return p
   end
end

mutable struct Storage
   index::Any
   bus_idx::Int
   s_P_max::Float64
   s_Q_max::Float64
   s_SOC_max::Float64
   s_SOC_min::Float64
   s_eff_char::Float64
   s_eff_dischar::Float64
   s_cap::Float64
   function Storage(index, bus_idx, s_P_max, s_S_max, s_SOC_max, s_SOC_min, s_eff_char, s_eff_dischar, s_cap)
      s = new()
      s.index  = index
      s.bus_idx = bus_idx
      s.s_P_max = s_P_max
      s.s_Q_max = sqrt(s_S_max^2 - s_P_max^2)
      s.s_SOC_max = s_SOC_max
      s.s_SOC_min = s_SOC_min
      s.s_eff_char = s_eff_char
      s.s_eff_dischar = s_eff_dischar
      s.s_cap = s_cap
      return s
   end
end

mutable struct Bus
   index::Any
   is_root::Bool
   d_P::Float64
   d_Q::Float64
   cosphi::Float64
   tanphi::Float64
   v_max::Float64
   v_min::Float64
   children::Vector{Int}
   ancestor::Vector{Int}
   generator::Generator
   wind::Wind
   pv::PV
   storage::Storage
   function Bus(index, d_P, d_Q, v_max, v_min)
      b = new()
      b.index = index
      b.is_root = false
      b.d_P = d_P
      b.d_Q = d_Q
      b.v_max = v_max
      b.v_min = v_min
      b.children = Int[]
      b.ancestor = Int[]
      cosphi = d_P/(sqrt(d_P^2 + d_Q^2))
      if isnan(cosphi)
        b.cosphi = 0
        b.tanphi = 0
      else
        b.cosphi = cosphi
        b.tanphi = tan(acos(cosphi))
      end
      return b
   end
end

mutable struct Line
   index::Any
   to_node::Int # the "to" node
   from_node::Int # the "from" node
   r::Float64 # the resistance value
   x::Float64 # the reactance value
   b::Float64 # the susceptance value
   s_max::Float64 # the capacity of the line
   function Line(index, to_node, from_node, r, x, s_max)
      l = new()
      l.index = index
      l.to_node = to_node
      l.from_node = from_node
      l.r = r
      l.x = x
      l.b = (x/(r^2 + x^2))
      l.s_max = s_max
      return l
   end
end

function load_case_data(;datafile = "")

# READ RAW DATA

@info "Reading Data"

if datafile == ""
  data_dir = DATA_DIR
else
  data_dir = datafile
end

data_path = dirname(pwd())*"/gym_opf_env/opf_envs/envs/data"

nodes_raw = CSV.read(data_path*"/network_data/$data_dir/nodes.csv", DataFrame)
sum(nonunique(nodes_raw, :index)) != 0 ? warn("Ambiguous Node Indices") : nothing

lines_raw = CSV.read(data_path*"/network_data/$data_dir/lines.csv", DataFrame)
sum(nonunique(lines_raw, :index)) != 0  ? warn("Ambiguous Line Indices") : nothing

generators_raw = CSV.read(data_path*"/network_data/$data_dir/generators.csv", DataFrame)
sum(nonunique(generators_raw, :index)) != 0 ? warn("Ambiguous Generator Indices") : nothing

windturbines_raw = CSV.read(data_path*"/network_data/$data_dir/windturbines.csv", DataFrame)
sum(nonunique(windturbines_raw, :index)) != 0 ? warn("Ambiguous Wind Turbine Indices") : nothing

pvs_raw = CSV.read(data_path*"/network_data/$data_dir/pvs.csv", DataFrame)
sum(nonunique(pvs_raw, :index)) != 0 ? warn("Ambiguous PV Indices") : nothing

storages_raw = CSV.read(data_path*"/network_data/$data_dir/storages.csv", DataFrame)
sum(nonunique(storages_raw, :index)) != 0 ? warn("Ambiguous Storage Indices") : nothing

# Base values

Zbase = 1
Vbase = 4160
Sbase = (Vbase^2)/Zbase
Cbase = 800

# PREPARING MODEL DATA

buses = Dict()
for n in 1:nrow(nodes_raw)
    index = nodes_raw[n, :index]
    d_P = 1000*nodes_raw[n, :d_P]/Sbase
    d_Q = 1000*nodes_raw[n, :d_Q]/Sbase
    v_max = nodes_raw[n, :v_max]
    v_min = nodes_raw[n, :v_min]
    newb = Bus(index, d_P, d_Q, v_max, v_min)
    buses[newb.index] = newb
end

lines = Dict()
for l in 1:nrow(lines_raw)
    index = lines_raw[l, :index]
    from_node = lines_raw[l, :from_node]
    to_node = lines_raw[l, :to_node]
    r = lines_raw[l, :r]/Zbase
    x = lines_raw[l, :x]/Zbase
    s_max = 1000*lines_raw[l, :s_max]/Sbase
    newl = Line(index, to_node, from_node, r, x, s_max)

    push!(buses[newl.from_node].children, newl.to_node)
    push!(buses[newl.to_node].ancestor, newl.from_node)

    lines[newl.index] = newl
end

# Check topology
r = 0
root_bus = 0
for b in keys(buses)
    l = length(buses[b].ancestor)
    if l > 1
        warn("Network not Radial (Bus $(buses[b].index))")
        #println("Network not Radial")
    elseif l == 0
        buses[b].is_root = true
        root_bus = b
        r += 1
    end
end
if r == 0
    warn("No root detected")
    root_bus = 0
elseif r > 1
    #warn("More than one root detected")
    println("More than one root detected")
end

generators = Dict()
for g in 1:nrow(generators_raw)
    index = generators_raw[g, :index]
    bus_idx = generators_raw[g, :node]
    g_P_max = 1000*generators_raw[g, :p_max]/Sbase
    g_S_max = 1000*generators_raw[g, :s_max]/Sbase
    cost = generators_raw[g, :cost]
    newg = Generator(index, bus_idx, g_P_max, g_S_max, cost)

    buses[newg.bus_idx].generator = newg

    generators[newg.index] = newg
end

windturbines = Dict()
for w in 1:nrow(windturbines_raw)
    index = windturbines_raw[w, :index]
    bus_idx = windturbines_raw[w, :node]
    w_P_max = 1000*windturbines_raw[w, :p_max]/Sbase
    w_S_max = 1000*windturbines_raw[w, :s_max]/Sbase
    neww = Wind(index, bus_idx, w_P_max, w_S_max)

    buses[neww.bus_idx].wind = neww

    windturbines[neww.index] = neww
end

pvs = Dict()
for p in 1:nrow(pvs_raw)
    index = pvs_raw[p, :index]
    bus_idx = pvs_raw[p, :node]
    p_P_max = 1000*pvs_raw[p, :p_max]/Sbase
    p_S_max = 1000*pvs_raw[p, :s_max]/Sbase
    newp = PV(index, bus_idx, p_P_max, p_S_max)

    buses[newp.bus_idx].pv = newp

    pvs[newp.index] = newp
end

storages = Dict()
for s in 1:nrow(storages_raw)
    index = storages_raw[s, :index]
    bus_idx = storages_raw[s, :node]
    s_P_max = 1000*storages_raw[s, :p_max]/Sbase
    s_S_max = 1000*storages_raw[s, :s_max]/Sbase
    s_SOC_max = storages_raw[s, :SOC_max]
    s_SOC_min = storages_raw[s, :SOC_min]
    s_eff_char = storages_raw[s, :eff_char]
    s_eff_dischar = storages_raw[s, :eff_dischar]
    s_cap = storages_raw[s, :capacity]/Cbase
    news = Storage(index, bus_idx, s_P_max, s_S_max, s_SOC_max, s_SOC_min, s_eff_char, s_eff_dischar, s_cap)

    buses[news.bus_idx].storage = news

    storages[news.index] = news
end

#info("Done preparing Data")
@info "Done preparing Data"
return buses, lines, generators, windturbines, pvs, storages
end
