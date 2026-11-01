# Group `examples`: the exported examples, and the plotter when the GUI tests are on.

@testset "Examples" begin
    @sim advection_2d
    @sim burgers_kinetic
    @sim bgk_1d_sod
    if RUN_GUI_TESTS
        @plot euler_1d_sod
        reset_plotter!()
    end
end
