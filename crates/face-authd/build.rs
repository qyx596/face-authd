fn main() {
    let opencv = pkg_config::Config::new()
        .atleast_version("4.8")
        .cargo_metadata(false)
        .probe("opencv4")
        .or_else(|_| {
            pkg_config::Config::new()
                .atleast_version("5.0")
                .cargo_metadata(false)
                .probe("opencv5")
        })
        .expect("OpenCV >= 4.8 development files required (pkg-config opencv4/opencv5)");
    let mut build = cc::Build::new();
    build.cpp(true).file("src/face_backend.cpp").std("c++17");
    for include in opencv.include_paths {
        build.include(include);
    }
    build.compile("face_backend");
    for path in opencv.link_paths {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
    for lib in [
        "opencv_objdetect",
        "opencv_dnn",
        "opencv_imgproc",
        "opencv_core",
    ] {
        println!("cargo:rustc-link-lib={lib}");
    }
    println!("cargo:rerun-if-changed=src/face_backend.cpp");
    println!("cargo:rerun-if-changed=src/face_backend.h");
}
