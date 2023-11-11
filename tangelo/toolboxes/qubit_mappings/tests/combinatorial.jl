
import Base.Threads.@spawn

using JLD2
using ProgressBars


const ZERO_TOLERANCE::Float64 = 1e-8
const BIG_N_QUBITS::Int = 7


function len_tuple(tuple)
    i = 0
    for _ in tuple
        i += 1
    end
    return i
end


function map_two_by_two_matrix(a, b, c, d)
    """ Matrix(a b // c d) to 1/2[ (a+d) I + (b+c) X + (b-c) iY + (a-d) Z ] """
    return Dict{Int, Complex}(0 => 0.5*(a+d), 1 => 0.5*(b+c), 2 => 0.5*(a-d), 3 => 0.5im*(b-c))
end


function tensor_product_pauli_dicts(pauli_op_a::Dict{Int, Complex}, pauli_op_b::Dict{Int, Complex})
    """Only tensor product (no consideration of phase shifts on operation on the
    same qubits).
    """
    pauli_product = Dict{Int, Complex}()
    for (term_a, coeff_a) in pauli_op_a
        for (term_b, coeff_b) in pauli_op_b
            pauli_product[xor(term_a, term_b)] = coeff_a * coeff_b
        end
    end

    return pauli_product
end


function recursive_mapping(M)
    n_rows, n_cols = size(M)
    @assert n_rows == n_cols

    if n_rows == 2
        # Indices starts at 1, remember?
        return map_two_by_two_matrix(M[1, 1], M[1, 2], M[2, 1], M[2, 2])
    else
        n_qubits::Int = log2(n_rows)

        pivr = n_rows ÷ 2
        pivc = n_cols ÷ 2

        shift_x::Int = 2*(n_qubits - 1)
        shift_z::Int = shift_x + 1

        zero::Int = 0
        one::Int = 1

        # 1/2 (I +- Z)
        z_op = zero | (one << shift_z)
        i_plus_z = Dict{Int, Complex}(0 => 0.5, z_op => 0.5)
        i_minus_z = Dict{Int, Complex}(0 => 0.5, z_op => -0.5)

        # 1/2 (X +- iY)
        x_op = zero | (one << shift_x)
        y_op = zero | (one << shift_x)
        y_op = y_op | (one << shift_z)

        x_plus_iy = Dict{Int, Complex}(x_op => 0.5, y_op => 0.5im)
        x_minus_iy = Dict{Int, Complex}(x_op => 0.5, y_op => -0.5im)

        if n_qubits > BIG_N_QUBITS
            M_00 = @spawn tensor_product_pauli_dicts(recursive_mapping(M[1:pivr, 1:pivc]), i_plus_z)
            M_11 = @spawn tensor_product_pauli_dicts(recursive_mapping(M[pivr+1:n_rows, pivc+1:n_cols]), i_minus_z)
            M_01 = @spawn tensor_product_pauli_dicts(recursive_mapping(M[1:pivr, pivc+1:n_cols]), x_plus_iy)
            M_10 = @spawn tensor_product_pauli_dicts(recursive_mapping(M[pivr+1:n_rows, 1:pivc]), x_minus_iy)
            return mergewith(+, fetch(M_00), fetch(M_01), fetch(M_10), fetch(M_11))
        else
            M_00 = tensor_product_pauli_dicts(recursive_mapping(M[1:pivr, 1:pivc]), i_plus_z)
            M_11 = tensor_product_pauli_dicts(recursive_mapping(M[pivr+1:n_rows, pivc+1:n_cols]), i_minus_z)
            M_01 = tensor_product_pauli_dicts(recursive_mapping(M[1:pivr, pivc+1:n_cols]), x_plus_iy)
            M_10 = tensor_product_pauli_dicts(recursive_mapping(M[pivr+1:n_rows, 1:pivc]), x_minus_iy)
            return mergewith(+, M_00, M_01, M_10, M_11)
        end
    end
end


function int_to_tuple(integer, n_qubits)

    term = []
    for i in 1:n_qubits

        shift_x = 2*(i-1)
        shift_z = shift_x+1
        one::Bool = 1

        x_term = (integer & (one << shift_x)) >> shift_x
        z_term = (integer & (one << shift_z)) >> shift_z

        if x_term == 0 && z_term == 0
            continue
        elseif x_term == 1 && z_term == 0
            push!(term , (i-1, "X"))
        elseif x_term == 0 && z_term == 1
            push!(term , (i-1, "Z"))
        else
            push!(term , (i-1, "Y"))
        end
    end

    return tuple(term...)
end


function one_body_op_on_state(op::NTuple{2, Tuple{Int, Bool}}, state_in::Tuple{Vararg{Int}})

    # Convert tuple to array (we want it to be mutable).
    state = collect(state_in)

    # Unpack the creation and annhilation operators.
    creation_op, annhilation_op = op
    creation_qubit, creation_dagger = creation_op
    annhilation_qubit, annhilation_dagger = annhilation_op

    # annhilation logics on the state.
    if annhilation_qubit in state
        deleteat!(state, findall(x->x==annhilation_qubit, state))
    else
        return (), 0
    end

    # Creation logics on the state.
    if !(creation_qubit in state)
        push!(state, creation_qubit)
    else
        return (), 0
    end

    # Compute the phase shift.
    if annhilation_qubit > creation_qubit
        d = length(filter(qubit -> creation_qubit < qubit < annhilation_qubit, state))
    elseif annhilation_qubit < creation_qubit
        d = length(filter(qubit -> annhilation_qubit < qubit < creation_qubit, state))
    else
        d = 0
    end

    return Tuple(sort(state)), (-1)^d
end

function get_qubit_op_dict(ferm_op, basis_set, n_qubits)

    # Specify the data types in the dictionaries.
    ferm_op = convert(Dict{Tuple{Vararg{Tuple{Int, Bool}}}, Float64}, ferm_op)
    basis_set = convert(Dict{Tuple{Vararg{Int16}}, Int}, basis_set)
    n_qubits = convert(Int, n_qubits)

    # Dictionary of terms to be output.
    quop_matrix = Dict{Tuple{Int, Int}, Complex}()

    # Handling the constant term.
    cte = 0.
    if haskey(ferm_op, ())
        cte = ferm_op[()]
        delete!(ferm_op, ())
    end

    n_terms = length(ferm_op)
    n_basis = length(basis_set)
    confs = Tuple{Vararg{Int}}[]
    ints = Int[]
    for (conf, unique_int) in basis_set
        push!(confs, conf)
        push!(ints, unique_int)
    end
    max_int = maximum(ints)

    println("Change to a configuration basis (build quop_matrix)")
    t = @elapsed begin
    Threads.@threads :static for i in 1:n_basis

        thread_dict = Dict{Tuple{Int, Int}, Complex}()
        conf = confs[i]
        unique_int = ints[i]

        filtered_ferm_op = filter(((k, v),) -> k[end][1] in conf, ferm_op)
        for (term, coeff) in filtered_ferm_op
            new_state, phase = one_body_op_on_state(term[end-1:end], conf)

            if (len_tuple(term) == 4) && !isempty(new_state)
                new_state, phase_two = one_body_op_on_state(term[1:2], new_state)
                phase *= phase_two
            end

            if isempty(new_state)
                continue
            end

            new_unique_int = basis_set[new_state]
            # Julia indices begin at 1.
            thread_dict[(unique_int + 1, new_unique_int + 1)] = get(thread_dict, (unique_int + 1, new_unique_int + 1), 0.) + (phase*coeff)
        end

        mergewith!(+, quop_matrix, thread_dict)

    end
    end
    println("[", t, " s elapsed ]")
    println("Number of terms : ", length(quop_matrix))
    return quop_matrix

    t = @elapsed begin
    println("Conversion of matrix into a qubit operator")
    quop_ints = recursive_mapping(quop_matrix)
    end
    println("[", t, " s elapsed ]")

    t = @elapsed begin
        quop = Dict{Any, Complex}()
        println("Conversion of int terms to tuples")
        for (term, coeff) in quop_ints
            if abs(coeff) < ZERO_TOLERANCE
                continue
            end
            quop[int_to_tuple(term, n_qubits)] = coeff
        end
        quop[()] += cte
    end
    println("[", t, " s elapsed ]")

    println("Returning result to python context")
    return quop
end

function get_qubit_op(ferm_op, basis_set, n_qubits)

    # Specify the data types in the dictionaries.
    ferm_op = convert(Dict{Tuple{Vararg{Tuple{Int, Bool}}}, Float64}, ferm_op)
    basis_set = convert(Dict{Tuple{Vararg{Int16}}, Int}, basis_set)
    n_qubits = convert(Int, n_qubits)

    # Dictionary of terms to be output.
    quop_matrix = zeros(Complex, 2^n_qubits, 2^n_qubits)

    # Handling the constant term.
    cte = 0.
    if haskey(ferm_op, ())
        cte = ferm_op[()]
        delete!(ferm_op, ())
    end

    n_terms = length(ferm_op)
    n_basis = length(basis_set)
    confs = Tuple{Vararg{Int}}[]
    ints = Int[]
    for (conf, unique_int) in basis_set
        push!(confs, conf)
        push!(ints, unique_int)
    end
    max_int = maximum(ints)

    println("Change to a configuration basis (build quop_matrix)")
    t = @elapsed begin
    Threads.@threads :static for i in 1:n_basis

        conf = confs[i]
        unique_int = ints[i]

        filtered_ferm_op = filter(((k, v),) -> k[end][1] in conf, ferm_op)
        for (term, coeff) in filtered_ferm_op
            new_state, phase = one_body_op_on_state(term[end-1:end], conf)

            if (len_tuple(term) == 4) && !isempty(new_state)
                new_state, phase_two = one_body_op_on_state(term[1:2], new_state)
                phase *= phase_two
            end

            if isempty(new_state)
                continue
            end

            new_unique_int = basis_set[new_state]

            # Julia indices begin at 1.
            quop_matrix[unique_int + 1, new_unique_int + 1] += (phase*coeff)
        end
    end
    end
    println("[", t, " s elapsed ]")

    t = @elapsed begin
    println("Conversion of matrix into a qubit operator")
    quop_ints = recursive_mapping(quop_matrix)
    end
    println("[", t, " s elapsed ]")

    t = @elapsed begin
        quop = Dict{Any, Complex}()
        println("Conversion of int terms to tuples")
        for (term, coeff) in quop_ints
            if abs(coeff) < ZERO_TOLERANCE
                continue
            end
            quop[int_to_tuple(term, n_qubits)] = coeff
        end
        quop[()] += cte
    end
    println("[", t, " s elapsed ]")

    println("Returning result to python context")
    return quop
end