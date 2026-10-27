set(TARGET agent-node)
# kcp: reliable delivery over UDP for direct connections between nodes (p2p.h), MIT, vendored unmodified
add_executable(${TARGET} agent-node.cpp kcp/ikcp.c)
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
target_link_libraries(${TARGET} PRIVATE llama-common llama cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
# cpp-httplib brings OpenSSL / BoringSSL for TLS and access tokens
if (WIN32)
    target_link_libraries(${TARGET} PRIVATE ws2_32 shell32 advapi32)
endif()
target_compile_features(${TARGET} PRIVATE cxx_std_17)
# vendored as is: its warnings are not ours to fix
set_source_files_properties(kcp/ikcp.c PROPERTIES COMPILE_OPTIONS "$<IF:$<C_COMPILER_ID:MSVC>,/w,-w>")
