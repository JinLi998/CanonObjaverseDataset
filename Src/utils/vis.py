


def vis_pytorch3d_mesh(mesh):
    from pytorch3d.vis.plotly_vis import plot_scene
    # Render the plotly figure
    fig = plot_scene({
        "subplot1": {
            "cow_mesh": mesh
        }
    })
    fig.show()